import torch
from botorch.test_functions import Hartmann, Cosine8, Ackley
from botorch.models import ModelListGP, FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import PosteriorMean, AnalyticAcquisitionFunction
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning
import warnings
import numpy as np
from torch import Tensor
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.fit import fit_gpytorch_model
from gpytorch.kernels import ScaleKernel, RFFKernel

from botorch.utils.transforms import t_batch_mode_transform

from contextlib import ExitStack
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize
import gpytorch.settings as gpts

from copy import deepcopy
from botorch.generation import MaxPosteriorSampling


def truth(X):
    return blackbox(X*Normalize_x)/Normalize_y

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

        noise = NOISE_SE * torch.ones_like(self.local_y) #* (1+2/np.sqrt(i+1))
        self.model = FixedNoiseGP(self.local_x, self.local_y, noise).to(device)

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model).to(device)
        fit_gpytorch_model(self.mll)

        self.optimized_param = deepcopy(self.model.state_dict())
        self.lengthscale = self.model.covar_module.base_kernel.lengthscale
        self.outputscale = self.model.covar_module.outputscale
        # self.obs_noise = self.model.likelihood.noise_covar.noise
        self.obs_noise = NOISE_SE
        self.rff_kernel = RFFKernel(ard_num_dims=dim, num_samples=M).to(dtype=self.lengthscale.dtype, device=self.lengthscale.device)
        self.BOD()
    
    def sample_w(self):
        Phi = self.outputscale.sqrt() * self.rff_kernel._featurize(self.local_x, normalize=True)

        Sigma_t = (Phi.T @ Phi) + self.obs_noise * torch.eye(2*M).cuda().double()
        Sigma_t_inv = torch.linalg.inv(Sigma_t)
        nu_t = ((Sigma_t_inv @ Phi.T) @ self.local_y.reshape(-1, 1))
        
        w_sample = torch.distributions.MultivariateNormal(torch.squeeze(nu_t), self.obs_noise * Sigma_t_inv).sample()
        return w_sample

    
    def predict(self, x, w = None):     
        if w == None:
            w = self.sample_w()
        features = self.outputscale.sqrt() * self.rff_kernel._featurize(x, normalize=True)
        features = features.reshape(-1,2*M)
        
        f_value = (w @ features.T).squeeze()
        
        return f_value  
    
    def propose(self):
        criteria = PosteriorMean(self.model)
        candidates, value = optimize_acqf(
            acq_function=criteria,
            bounds=bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": BATCH_LIMIT, "maxiter": MAX_ITR},
        )
        self.best_design = candidates
        self.local_best_y = value.item()
    
    
    def get_next_design(self, n_candidates = 10000): 
        # Draw samples on a Sobol sequence
        sobol = SobolEngine(self.local_x.shape[-1], scramble=True)
        X_cand = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        X_cand = unnormalize(X_cand, bounds)
        
        # Thompson sample
        with ExitStack() as es:
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
            candidates = thompson_sampling(X_cand, num_samples=1)
            
        return candidates.detach()
        
    def get_next_observation(self):
        new_x = self.next_sample
        new_y = self.observation(new_x)
        self.local_x = torch.cat([self.local_x, new_x])
        self.local_y = torch.cat([self.local_y, new_y])
        self.update_model()
    
    def regret(self, X):
        self.simple_regret = 3.32237 - truth(X + self.heter).unsqueeze(-1)
    
    
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
        randn_weights = sample_matern_weight()
        self.weight_list = torch.tensor([]).to(device)
        for agent in self.agents:
            agent.rff_kernel.randn_weights = randn_weights
            self.weight_list = torch.cat([self.weight_list, agent.sample_w().unsqueeze(0)])
        for agent in self.agents:
            agent.borrow_weights = self.weight_list
            
    def researching(self):
        for agent in self.agents:
            
            if torch.rand(1).item() <= p:
                agent.next_sample = agent.get_next_design(N_TS_samples)
            else:
                teacher_ind = int((torch.rand(1)*N).floor())
                teacher = self.agents[teacher_ind]
                agent.next_sample = teacher.get_next_design(N_TS_samples)
                
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
        

def sample_matern_weight(p_number=2):
    df = 2 * (p_number + 0.5)
    y = torch.randn(dim, M, dtype=dtype, device=device )
    u = torch.distributions.chi2.Chi2(df).sample((M,)).to(dtype=dtype, device=device)
    randn_weights = (y * torch.sqrt(df / u))
    return randn_weights


if __name__ == "__main__":
    print("cos-FTS")
    
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
    # for RFF
    M = 1000

    # for GP
    Normalize_x = 2
    Normalize_y = 1
    
    # Setting parameters
    blackbox = Cosine8()
    NOISE_SE = 0.1/Normalize_y
    dim = blackbox.dim
    bounds = torch.tensor([[-1.0] * dim, [1.0] * dim], device=device, dtype=dtype)/Normalize_x
    heter = 0.1
    initial_samples = 10
    N = 16
    
    # Contorlling parameters (affects peformance)
    MAX_GROUP_SIZE = 4 #change to 1 for no collabration
    Com_itr = 1
    c = 2
    beta = 4 #.sqrt()
    
    BOD_table = []
    simple_table = []
    for rep in range(5):
        seed = 200+rep
        torch.manual_seed(seed)
        itr = 1
        server = community(N_agents = N)
        simple_record = []
        BOD_record = []
        
        _, BOD_reward, current_reward = server.report()
        simple_record.append(current_reward)
        BOD_record.append(BOD_reward)
        
        for itr in range(49): 
            itr += 2
            p = 1 - 1/np.sqrt(itr)
            server.conference()
            server.researching()
            
            _, BOD_reward, current_reward = server.report()
            simple_record.append(current_reward)
            BOD_record.append(BOD_reward)
        
        BOD_table.append(BOD_record)
        simple_table.append(simple_record)
    
    import pandas as pd
    pd.DataFrame(BOD_table).T.to_excel("cos_fts_bod.xlsx", index=False, engine='openpyxl')
    pd.DataFrame(simple_table).T.to_excel("cos_fts_int.xlsx", index=False, engine='openpyxl')
        
        
        


    
        


    