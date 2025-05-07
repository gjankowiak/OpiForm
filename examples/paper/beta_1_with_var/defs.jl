dir_label = "paper/beta_1_with_var"

target_communities = 3

N_micro = 1000
N_mfl = 303
max_iter = 250000

nmin = round(Int64, N_micro / 5)
nmax = round(Int64, N_micro / 2.5)
k_mean = 15 # mean degree
k_max = 45  # maximum degree

#β_σ²s = [1e-3; 4e-3; 8e-3; 1.2e-2]
β_σ²s = [1e-3; 4e-3; 1.2e-2]
β_σ² = β_σ²s[1]
μs = [1e-3; 5e-3; 1e-2; range(5e-2, 5e-1, 7)]
μs = [1e-3; 1e-2; 1e-1; 0.5]

function get_reference_dir(μ)
  return "results/$(dir_label)/N_micro=$(N_micro)/reference/μ=$μ"
end

