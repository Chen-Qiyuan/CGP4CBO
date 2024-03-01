import torch
from torch import Tensor

from botorch.test_functions import Hartmann, Cosine8, Levy, Ackley
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.acquisition import PosteriorMean, AnalyticAcquisitionFunction, UpperConfidenceBound, AcquisitionFunction
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning
from botorch.models.gp_regression import FixedNoiseGP
from botorch.sampling.normal import SobolQMCNormalSampler, IIDNormalSampler
from botorch.fit import fit_gpytorch_model
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.mlls import ExactMarginalLogLikelihood
import random
from gpytorch.kernels import ScaleKernel, RBFKernel
import pandas as pd

import warnings
import numpy as np
from copy import deepcopy


def truth(X):
    return blackbox(X*Normalize_x)/Normalize_y

class LowerConfidenceBound(AnalyticAcquisitionFunction):

    def __init__(
        self,
        model,
        beta,
        posterior_transform = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:

        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mean, sigma = self._mean_and_sigma(X)
        return (mean if self.maximize else -mean) - self.beta * sigma


class LUCB(AnalyticAcquisitionFunction):

    def __init__(
        self,
        model_list,
        X_observed,
        beta = 2):
        
        # The fantasy model will operate in batch mode
        self.beta = torch.tensor(beta)
        Fantasy_X = torch.tensor([]).to(device)
        Fantasy_Y = torch.tensor([]).to(device)
        Fantasy_Y_var = torch.tensor([]).to(device)
        for model in model_list:
            train_X = model.train_inputs[0].unsqueeze(0)
            train_y = model.train_targets.unsqueeze(0)
            noise = model.likelihood.noise_covar.noise.unsqueeze(0)
            Fantasy_X = torch.cat([Fantasy_X, train_X ])
            Fantasy_Y = torch.cat([Fantasy_Y, train_y])
            Fantasy_Y_var = torch.cat([Fantasy_Y_var, noise])

        fantasy_model = FixedNoiseGP(
                    train_X = torch.zeros(1,bounds.shape[1]).to(dtype),
                    train_Y = torch.zeros(1,1).to(dtype),
                    train_Yvar = torch.ones(1,1).to(dtype),
                ).to(device)

        fantasy_model.set_train_data( inputs = Fantasy_X, targets = Fantasy_Y, strict=False )
        fantasy_model.likelihood.noise_covar.noise = Fantasy_Y_var
        fantasy_model.load_state_dict( model.state_dict() )
        
        super().__init__(model=fantasy_model)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mean, sigma = self._mean_and_sigma(X.squeeze())
        conditioned_mean = mean.mean(dim=0)
        conditioned_sigma = torch.square(sigma[0]) + mean.var(dim=0)
        return conditioned_mean + self.beta.sqrt() * conditioned_sigma.sqrt()




class agent():
    
    def __init__(self, id):
        self.cube_lb , self.cube_length = None, None
        self.pseudo_x, self.pseudo_y, self.pseudo_noise = Tensor([]).to(device), Tensor([]).to(device), Tensor([]).to(device)
        self.have_fantasy = False
        self.simple_regret = 0
        self.id = id
        if heter > 0:
            self.heter = torch.tensor([torch.inf])
            while torch.linalg.norm(self.heter) > heter:
                self.heter = (torch.rand(1,bounds.shape[1]).to(device)-0.5) * heter
        else:
            self.heter = 0
            
        
    def BOD_reward(self):
        return truth(self.best_design + self.heter) * Normalize_y
    
    def observation(self, X):
        exact_y = truth(X + self.heter).unsqueeze(-1)
        self.current_reward = exact_y * Normalize_y
        return exact_y + NOISE_SE * torch.randn_like(exact_y)
    
    def generate_initial_data(self, n = 1):
        # generate training data        
        self.local_x = torch.rand(n, bounds.shape[1]) * self.cube_length + self.cube_lb
        self.local_x = self.local_x.type(dtype).to(device)
        self.local_y = self.observation(self.local_x)
        self.update_model()
    
    def update_model(self):      
        noise = NOISE_SE * torch.ones_like(self.local_y) 
        self.model = FixedNoiseGP(self.local_x, self.local_y, noise).to(device)
        # self.model = SingleTaskGP(self.local_x, self.local_y).to(device)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model).to(device)
        fit_gpytorch_model(self.mll)
        self.optimized_param = deepcopy(self.model.state_dict())
        self.BOD()
        


    def gen_fantasy_models(self, pseudo_x, num_fantasies = 20):
        
        self.pseudo_x = pseudo_x.clone()

        if self.pseudo_x.shape[0]==0:
            self.have_fantasy = False
            self.truncate_models = None 
            return
        
        else: 
            num_samples = int(1e5)
            sampler = IIDNormalSampler(int(num_samples))
            posterior = self.model.posterior(self.pseudo_x)
            samples = sampler(posterior)
            matching = (samples > self.best_value)
            
            patterns, counts = torch.unique(matching, dim=0, return_counts=True)
            patterns = patterns[counts>MIN_SAMPLE]
            
            value , indices = patterns.sum(dim=1).squeeze().sort()
            
            if indices.numel() == 1:
                print("Automatically excluded all samples")
                self.have_fantasy = False
                self.truncate_models = None 
                return
            
            auto_select = patterns[indices[-1]].reshape(-1)
            
            self.pseudo_x = pseudo_x[auto_select].clone()

            auto_select_exp = auto_select.expand_as(matching.squeeze())
            selected_indices = (matching.squeeze() == auto_select_exp).reshape(num_samples,-1).all(dim=1)
            selected_samples = samples[selected_indices][:,auto_select,:][:num_fantasies]

            model_list = []
            
            for fantasy_y in selected_samples:
                train_x = torch.cat([self.local_x, self.pseudo_x])
                train_y = torch.cat([self.local_y, fantasy_y.reshape(-1,1)])
                noise = torch.cat([NOISE_SE * torch.ones_like(self.local_y), small_nug * torch.ones_like(fantasy_y.reshape(-1,1))])
                
                truncate_model = FixedNoiseGP(train_x, train_y, noise).to(device)

                truncate_model.load_state_dict(self.optimized_param)
                model_list.append(truncate_model)
            
            self.truncate_models = model_list
            self.have_fantasy = True
            
            if value[-1] < pseudo_x.shape[0]:
                print("Exclude designs:", 
                      (pseudo_x.shape[0] - value[-1]).item(),
                      "")
            return

    def get_next_design(self):
        if self.have_fantasy:
            # define acquisition function
            acq_func = LUCB(
                model_list = self.truncate_models, 
                X_observed = self.local_x,
                beta= beta
            )
            # optimize acquisition function
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                options={"batch_limit": BATCH_LIMIT, "maxiter": MAX_ITR},
            )
            
            # observe new values 
            self.next_sample = candidates.detach()
        else:
            acq_func = UpperConfidenceBound(
                model = self.model, 
                beta = beta,
            )
                    
            # optimize acquisition function
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                options={"batch_limit": BATCH_LIMIT, "maxiter": MAX_ITR},
            )
            
            # observe new values 
            self.next_sample = candidates.detach()
        
    def get_next_observation(self):
        new_x = self.next_sample
        new_y = self.observation(new_x)
        self.local_x = torch.cat([self.local_x, new_x])
        self.local_y = torch.cat([self.local_y, new_y])
        self.update_model()
    
    def regret(self, X):
        self.simple_regret = 3.32237 - truth(X + self.heter).unsqueeze(-1)
    
    def propose(self):        
        criteria = LowerConfidenceBound( self.model, 2)
        candidates, value = optimize_acqf(
            acq_function=criteria,
            bounds=bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": BATCH_LIMIT, "maxiter": MAX_ITR},
        )
        
        self.proposed_design = candidates
        self.proposed_value = value.item()
    
    def BOD(self):
        criteria = PosteriorMean(self.model)
        candidates, value = optimize_acqf(
            acq_function=criteria,
            bounds=bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": BATCH_LIMIT, "maxiter": MAX_ITR},
        )

        self.best_value = value.item()
        self.best_design = candidates
        
class community():
    
    def __init__(self, N_agents = 1):
        self.N_agents = N_agents
        self.agent_index = range(N_agents)
        self.agents = [agent(i) for i in self.agent_index]
        self.equal_CLHS(initial_samples)
        self.get_initial_designs()
    
    def get_initial_designs(self):
        for agent in self.agents:
            agent.generate_initial_data(initial_samples)
            
    def equal_CLHS(self, initial_samples):
        
        for dim in range(bounds.shape[1]):
            lb, ub = bounds[:, dim]
            # divide_cube = torch.linspace(lb, ub, self.N_agents*initial_samples+1 )[:-1]
            divide_cube = torch.linspace(lb, ub, self.N_agents*initial_samples+1 )
            
            for i, agent_id in enumerate(torch.randperm(self.N_agents)):
                LHS_perm = torch.randperm(initial_samples)
                
                if self.agents[agent_id].cube_lb == None:
                    self.agents[agent_id].cube_lb = divide_cube[i:-1:self.N_agents][LHS_perm].unsqueeze(-1)
                    # self.agents[agent_id].cube_lb = divide_cube[(i)*initial_samples:(i+1)*initial_samples][LHS_perm].unsqueeze(-1)
                    self.agents[agent_id].cube_length = Tensor([(ub - lb)/self.N_agents/initial_samples])
                    
                else:
                    self.agents[agent_id].cube_lb = torch.hstack([self.agents[agent_id].cube_lb, 
                                                                  divide_cube[i:-1:self.N_agents][LHS_perm].unsqueeze(-1),
                                                                 # divide_cube[(i)*initial_samples:(i+1)*initial_samples][LHS_perm].unsqueeze(-1)
                                                                 ]
                                                                )
                    self.agents[agent_id].cube_length = torch.hstack([self.agents[agent_id].cube_length, 
                                                                 Tensor([(ub - lb)/(self.N_agents*initial_samples)])
                                                                 ]
                                                                )
    def conference(self):
        if (itr-1) % Com_itr == 0:
            self.grouping = self.random_grouping(self.agents)
        
        borrow_count = 0
        for group in self.grouping:
            best_preds = []
            proposal = torch.tensor([]).to(device)
            for agent in group:
                agent.propose()
                best_preds.append(agent.proposed_value)
                proposal = torch.cat( (proposal, agent.proposed_design) )
            
            best_preds = np.array(best_preds)
            
            for i, agent in enumerate(group):
                emulation = proposal[best_preds > agent.best_value + tau]
                agent.gen_fantasy_models(emulation, num_fantasies = MAX_SAMPLE)  
                borrow_count += agent.have_fantasy
        print(borrow_count)
            
    def researching(self):
        for agent in self.agents:
            agent.get_next_design()
            agent.get_next_observation()
    
    def report(self):
        current_reward = torch.tensor([]).to(device)
        bod_reward = torch.tensor([]).to(device)
        for agent in self.agents:
            bod_reward = torch.cat([bod_reward, agent.BOD_reward()])
            current_reward = torch.cat([current_reward, agent.current_reward])
        
        print("Iteration", itr , 
              "| Best BOD Reward", bod_reward.max().item(),
              "| Avg BOD Reward", bod_reward.mean().item(),
              "| Avg Current Reward", current_reward.mean().item(),
              )
        
        return bod_reward.max().item(), bod_reward.mean().item(), current_reward.mean().item()
    
    def random_grouping(self, input_list):
        random.shuffle(input_list)
        N_Groups = int(np.ceil(N / MAX_GROUP_SIZE))
        return [sublist.tolist() for sublist in np.array_split(input_list, N_Groups)]
        

if __name__ == "__main__":
    print("levy-CUCB")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Now using device", device)
    dtype = torch.double
    warnings.filterwarnings('ignore', category = BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category = RuntimeWarning)
    warnings.filterwarnings('ignore', category=InputDataWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    # Numerical parameters (only affects the precision)
    # for optimizing acquisition functions
    NUM_RESTARTS = 100
    RAW_SAMPLES = 1000
    BATCH_LIMIT = 200
    MAX_ITR = 50
    N_TS_samples = 10000
    # for collabration
    MAX_SAMPLE = 20
    MIN_SAMPLE = 5
    tau = 1e-4 # mimic a low-bandwidth setting (e.g., half precision)
    # for GP
    Normalize_x = 20
    Normalize_y = 100
    small_nug = 1e-4
    
    # Setting parameters
    blackbox = Levy(negate=True, dim=20)
    NOISE_SE = 0.1/Normalize_y
    dim = blackbox.dim
    bounds = torch.tensor([[-10] * dim, [10] * dim], device=device, dtype=dtype)/Normalize_x
    heter = 0.1
    initial_samples = 25
    N = 16
    
    # Contorlling parameters (affects peformance)
    MAX_GROUP_SIZE = 4 #change to 1 for no collabration
    Com_itr = 1
    c = 2
    beta = 4 #.sqrt()
    
    BOD_table = []
    simple_table = []
    for rep in range(5):
        seed = 300+rep
        torch.manual_seed(seed)
        random.seed(seed)
        itr = 1
        server = community(N_agents = N)
        simple_record = []
        BOD_record = []
        
        _, BOD_reward, current_reward = server.report()
        simple_record.append(current_reward)
        BOD_record.append(BOD_reward)
        
        for itr in range(49): 
            itr += 2
            server.conference()
            server.researching()
            
            _, BOD_reward, current_reward = server.report()
            simple_record.append(current_reward)
            BOD_record.append(BOD_reward)
        
        BOD_table.append(BOD_record)
        simple_table.append(simple_record)

    pd.DataFrame(BOD_table).T.to_excel("levy_cucb_bod.xlsx", index=False, engine='openpyxl')
    pd.DataFrame(simple_table).T.to_excel("levy_cucb_int.xlsx", index=False, engine='openpyxl')


    