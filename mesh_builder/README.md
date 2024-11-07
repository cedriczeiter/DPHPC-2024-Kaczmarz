# Building `mesh_builder`
In principle, `mesh_builder` can be built the same way as `kaczmarz` (see `README.md` in repo root). However, building `mesh_builder` is intentionally separate from building `kaczmarz` because it requires LehrFEM++. That might make the build long/demanding. We think it should be possible to build (and use) kaczmarz without building `mesh_builder` and so it is.

# Using `mesh_builder`
When built, the `mesh_builder` expects a json configuration file and outputs a text-based representation of an LSE. The configuration file should specify a boundary value problem and the outputted LSE is the discretization of that BVP. See the existing `configuration.json` file an example of what the configuration looks like. When using the mesh builder executable, the input configuration can either be directly provided on standard input or the filename can be given as the one and only command-line argument. The LSE is outputted on standard output (it is recommended to redirect that to a file). For example,

(configuration on standard input)
```
./mesh_builder < ../configuration.json > lse.txt
```

or

(configuration file as command-line argument)
```
./mesh_builder ../configuration.json > lse.txt
```

# Creating several meshes at once
The shell script `generate_meshes.sh` can be used to create linear systems of increasing complexity. At the moment, this script generates meshes for PDE 1 used to benchmark Kaczmarz solvers presented here (https://www.sciencedirect.com/science/article/pii/S0167819109001252#sec5). The LSEs are saved in the folder `generated_bvp_matrices`.