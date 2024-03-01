Numerical experiments can be replicated using the code here. 
Under each test function, one can find different method by refering to their file names.
To replicate our result, simply run the corresponding file under the folder.

CNEI: Constrained Gaussian Process with Noisy Expected Improvement (Implemented as Log NEI to prevent numerical issues)
CTS: Constrained Gaussian Process with Thompson Sampling
CUCB: Constrained Gaussian Process with Upper Confidence Bound
FTS: Federated Thompson Sampling
FTSDE: Federated Thompson Sampling with Distributed Exploration
NEI: Noisy Expected Improvement (Implemented as Log NEI to prevent numerical issues)
TS: Thompson Sampling
UCB: Upper Confidence Bound

The output of the results are included in the corresponding excel files for readability. 
The random seed are fixed for every file, so if one has the same machine setup as ours, the results should be exactly the same.

For other details on the algorithm. Please refer to the original paper. 