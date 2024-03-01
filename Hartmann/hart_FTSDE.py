import torch
from botorch.test_functions import Hartmann, Cosine8, Levy, Ackley
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
from botorch.optim.initializers import initialize_q_batch_nonneg

from copy import deepcopy
from botorch.generation.sampling import SamplingStrategy
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
class MaxPosteriorSampling(SamplingStrategy):
    def __init__(
        self,
        model,
        objective = None,
        posterior_transform = None,
        replacement: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        if objective is None:
            objective = IdentityMCObjective()
        elif not isinstance(objective, MCAcquisitionObjective):
            if posterior_transform is not None:
                assert False
            else:
                posterior_transform = ScalarizedPosteriorTransform(
                    weights=objective.weights, offset=objective.offset
                )
                objective = IdentityMCObjective()
        self.objective = objective
        self.posterior_transform = posterior_transform
        self.replacement = replacement
    
    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        posterior = self.model.posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=self.posterior_transform,
        )
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        return self.maximize_samples(X, samples, num_samples)


    def maximize_samples(self, X: Tensor, samples: Tensor, num_samples: int = 1):
        obj = self.objective(samples, X=X) 
        values, idcs = torch.max(obj, dim=-1)
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        return values[0].detach().reshape((-1,1)), torch.gather(Xe, -2, idcs)

def truth(X):
    return blackbox(X*Normalize_x)/Normalize_y

def optimize(f, subbound, sub_weights = None):

    # generate a large number of random q-batches
    Xraw = subbound[0] + (subbound[1] - subbound[0]) * torch.rand(RAW_SAMPLES, dim).to(device,dtype)
    Yraw = f(Xraw, sub_weights.data)  # evaluate the acquisition function on these q-batches
    
    X = initialize_q_batch_nonneg(Xraw, Yraw, NUM_RESTARTS).clone()
    
    X.requires_grad_(True);
    
    optimizer = torch.optim.Adam([X], lr=0.01)
    
    # run a basic optimization loop
    for i in range(MAX_ITR):
        optimizer.zero_grad()
        loss = - f(X, sub_weights.data).sum() 
        loss.backward()  # perform backward pass
        optimizer.step()  # take a step
        for j, (lb, ub) in enumerate(zip(*bounds)):
            X.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
        

    X_output = X[f(X, sub_weights.data).argmax()]
    
    return f(X_output, sub_weights.data) , X_output.detach()
        

class agent():
    
    def __init__(self):
        self.cube_lb , self.cube_length = None, None
        self.pseudo_x, self.pseudo_y, self.pseudo_noise = Tensor([]), Tensor([]), Tensor([])
        if heter > 0:
            self.heter = torch.tensor([torch.inf])
            while torch.linalg.norm(self.heter) > heter:
                self.heter = (torch.rand(1,dim).to(device)-0.5) * heter
        else:
            self.heter = 0
        
    def BOD_reward(self):
        return truth(self.best_design + self.heter) * Normalize_y
    
    def observation(self, X):
        exact_y = truth(X + self.heter).unsqueeze(-1)
        self.current_reward = exact_y * Normalize_y
        # print(self.current_reward)
        return exact_y + NOISE_SE * torch.randn_like(exact_y)
    
    def generate_initial_data(self, n = 1):
        # generate training data        
        self.local_x = torch.rand(n, dim).type(dtype).to(device)
        self.local_x = unnormalize(self.local_x, self.bounds)
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
        self.obs_noise = NOISE_SE
        self.rff_kernel = RFFKernel(ard_num_dims=dim, num_samples=M).to(dtype=self.lengthscale.dtype, device=self.lengthscale.device)
        self.rff_kernel.randn_weights = randn_weights
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
        features = self.outputscale.sqrt().data * self.rff_kernel._featurize(x, normalize=True)
        features = features.reshape(-1,2*M)
        
        f_value = (w @ features.T).squeeze()
        
        return f_value
    
        
    def col_sample(self):
        values_list, candidates_list = torch.tensor([]).to(device), torch.tensor([]).to(device)
        itr = int(N/4)
        for i in range(4):
            bias = [*range(itr*i,itr*(i+1))]
            
            a = 16
            T = (i+1)**2

            weights=torch.zeros(N).type(dtype).to(device)
            weights[bias] = 1
            weights = torch.exp((a*weights+1)/T)
            weights = weights/weights.sum()
            
            sub_weights = weights @ self.borrow_weights
            
            value , candidate = optimize(self.predict, subbounds[i], sub_weights)
            values_list = torch.cat([values_list, value.unsqueeze(0)])
            candidates_list = torch.cat([candidates_list, candidate.unsqueeze(0)])
      
        return candidates_list[values_list.argmax()].unsqueeze(0)
        
    
    def get_next_observation(self):
        new_x = self.next_sample
        new_y = self.observation(new_x)
        self.local_x = torch.cat([self.local_x, new_x])
        self.local_y = torch.cat([self.local_y, new_y])
        self.update_model()
    
    
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
            values, candidates = thompson_sampling(X_cand, num_samples=1)
            
        return candidates.detach()
    
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
        self.agents = [agent() for i in self.agent_index]
        self.divide()
        self.get_initial_designs()

    def get_initial_designs(self):     
        for agent in self.agents:
            agent.generate_initial_data(initial_samples)
               
    def divide(self):
        itr = int(self.N_agents/P)
        for i in range(itr):
            self.agents[i].bounds = subbounds[0]
            
            self.agents[i+itr].bounds = subbounds[1]
            
            self.agents[i+itr*2].bounds = subbounds[2]
            
            self.agents[i+itr*3].bounds = subbounds[3]
    
    def conference(self):
        self.weight_list = torch.tensor([]).to(device)
        for agent in self.agents:
            self.weight_list = torch.cat([self.weight_list, agent.sample_w().unsqueeze(0)])
        for agent in self.agents:
            agent.borrow_weights = self.weight_list
    
    def researching(self):

        for agent in self.agents:
            if torch.rand(1).item() <= p:
                agent.next_sample = agent.get_next_design(n_candidates=N_TS_samples)
            else:
                agent.next_sample = agent.col_sample()
        
        for agent in self.agents:
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

def divide_subbounds():     
    subbounds = []
    
    subbound1 = bounds.clone()
    subbound1[0,0] = subbound1[:,0].mean()
    subbound1[0,1] = subbound1[:,1].mean()
    subbounds.append(subbound1)
    
    subbound2 = bounds.clone()
    subbound2[1,0] = subbound2[:,0].mean()
    subbound2[1,1] = subbound2[:,1].mean()
    subbounds.append(subbound2)
    
    subbound3 = bounds.clone()
    subbound3[1,0] = subbound3[:,0].mean()
    subbound3[0,1] = subbound3[:,1].mean()
    subbounds.append(subbound3)
    
    subbound4 = bounds.clone()
    subbound4[0,0] = subbound4[:,0].mean()
    subbound4[1,1] = subbound4[:,1].mean()
    subbounds.append(subbound4)
    
    # print(subbounds)
    return subbounds


if __name__ == "__main__":
    print("hart6-FTSDE")
    
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
    M = 1000 # number of rff features
    P = 4 # number of divided regions
    
    # for GP
    Normalize_x = 1
    Normalize_y = 1
    
    # Setting parameters
    blackbox = Hartmann(negate=True)
    NOISE_SE = 0.1/Normalize_y
    dim = blackbox.dim
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], device=device, dtype=dtype)/Normalize_x
    heter = 0.1
    initial_samples = 10
    N = 16
    
    # Contorlling parameters (affects peformance)
    MAX_GROUP_SIZE = 4 #change to 1 for no collabration
    Com_itr = 1
    c = 2
    beta = 4 #.sqrt()

    subbounds = divide_subbounds()

    BOD_table = []
    simple_table = []
    for rep in range(5):
        seed = 100+rep
        torch.manual_seed(seed)
        itr = 1
        randn_weights = sample_matern_weight()
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
    pd.DataFrame(BOD_table).T.to_excel("hart_ftsde_bod.xlsx", index=False, engine='openpyxl')
    pd.DataFrame(simple_table).T.to_excel("hart_ftsde_int.xlsx", index=False, engine='openpyxl')
        
