# SKETCH

# create a JokerRun for this run

# create prior samples cache, store to file and store filename in DB

# ...

# I'll need to write a careful adaptive scheme for the rejection sampling. My
#   original idea was to do something like this:

n_req = 128 # number of samples requested
n_process = 64 * n_req
n_samples = 0
for n in range(maxiter):
    # grab n_process samples from prior cache

    # process, see how many remain

    n_survive = ...

    if n_survive > 1:
        n_samples += n_survive
        # save info on the samples that survived

    if n_samples >= n_req:
        break

    n_process *= 2

else: # hit maxiter
    # TODO: warning...

