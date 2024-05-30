function load_worker()
    println("Spawning Worker")
    worker_pid = addprocs(1)[1]
    f = @spawnat worker_pid include("Experiments/run_experiments_worker.jl")
    fetch(f)
    return worker_pid
end

function clear_channel(c)
    while isready(c)
        take!(c)
    end
end

const results_channel = RemoteChannel(()->Channel{Any}(10000), 1)
const status_channel = RemoteChannel(()->Channel{Any}(10000), 1)

function run_experiments(experiment_params::Vector{ExperimentParams})
    for experiment in experiment_params
        clear_channel(results_channel)
        clear_channel(status_channel)
        results = [("Workload", "QueryType", "QueryPath", "Runtime", "OptTime", "CompileTime", "Result", "Failed")]
        num_attempted, num_completed, num_correct, num_with_values, exp_finished = (0, 0, 0, 0, false)
        put!(status_channel, (num_attempted, num_completed, num_correct, num_with_values, exp_finished))
        worker_pid = load_worker()
        cur_query = 1
        while !exp_finished
            if worker_pid == -1
                worker_pid = load_worker()
            end
            f = @spawnat worker_pid attempt_experiment(experiment, cur_query, results_channel, status_channel)
            load_start = time()
            finished = false
            last_result = time()
            while !finished
                sleep(.01)
                if isready(results_channel)
                    push!(results, take!(results_channel))
                    cur_query += 1
                    last_result = time()
                    num_attempted, num_completed, num_correct, num_with_values, exp_finished = fetch(status_channel)
                    finished = exp_finished
                end
                if ((time()-last_result) > experiment.timeout) && (time() - load_start > 100)
                    println("REMOVING WORKER")
                    interrupt(worker_pid)
                    rmprocs(worker_pid)
                    num_attempted, num_completed, num_correct, num_with_values, exp_finished = take!(status_channel)
                    if !exp_finished
                        num_attempted += 1
                    end
                    put!(status_channel, (num_attempted, num_completed, num_correct, num_with_values, exp_finished))
                    cur_query += 1
                    worker_pid = -1
                    finished = true
                end
            end
        end
        println("Attempted Queries: ", num_attempted)
        println("Completed Queries: ", num_completed)
        println("Queries With Ground Truth: ", num_with_values)
        println("Correct Queries: ", num_correct)
        filename = "Experiments/Results/" * param_to_results_filename(experiment)
        writedlm(filename, results, ',')
    end
end

#=
A = Tensor(SparseList(Element(0.0)), fsprand(100_000, 0.001))
B = Tensor(SparseList(Element(0.0)), fsprand(100_000, 0.1))
C = Tensor(SparseList(Element(0.0)), fsprand(100_000, 0.1))

@btime begin
    A = $A
    B = $B
    s = Scalar(0.0)
    @finch begin
       for i = _
          s[] +=  A[i] * B[i]
       end
    end
    s[]
 end


@btime begin
    A = $A
    B = $B
    C = $C
    s = Scalar(0.0)
    @finch begin
       for i = _
          s[] +=  A[i] * B[i] * C[i]
       end
    end
    s[]
 end


@btime begin
    A = $A
    B = $B
    s = Scalar(0.0)
    @finch begin
       for i = _
          s[] +=  A[gallop(i)] + B[gallop(i)]
       end
    end
    s[]
 end


 @btime begin
     A = $A
     B = $B
     C = $C
     s = Scalar(0.0)
     @finch begin
        for i = _
           s[] += A[gallop(i)] + B[gallop(i)] + C[gallop(i)]
        end
     end
     s[]
  end
 =#
