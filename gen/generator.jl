using Pkg
using Pkg.Artifacts
using Clang.Generators
using Clang.Generators.JLLEnvs
using tblis_jll
using JuliaFormatter

cd(@__DIR__)

# headers
include_dir = normpath(joinpath(tblis_jll.artifact_dir, "include"))

tci_h = joinpath(include_dir, "tci.h")
@assert isfile(tci_h)
tblis_h = joinpath(include_dir, "tblis", "tblis.h")
@assert isfile(tblis_h)

# load common option
options = load_options(joinpath(@__DIR__, "generator.toml"))

# run generator for all platforms
for target in JLLEnvs.JLL_ENV_TRIPLES
    @info "processing $target"

    options["general"]["output_file_path"] = joinpath(@__DIR__, "..", "src", "lib",
                                                      "$target.jl")

    args = get_default_args(target)
    push!(args, "-I$include_dir")

    header_files = [tci_h, tblis_h]

    ctx = create_context(header_files, args, options)

    build!(ctx)

    path = options["general"]["output_file_path"]
    format_file(path, YASStyle())
end
