{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : 
let
  lib = nixpkgs.lib;
  pname = "jax";
  version = "0.4.34";
  fetchPypi = python.pkgs.fetchPypi;
  setuptools = build-system.setuptools;
  ml-dtypes = dependencies.ml-dtypes;
  numpy = dependencies.numpy;
  scipy = dependencies.scipy;
  opt-einsum = dependencies.opt-einsum;
  jaxlib = dependencies.jaxlib;
in
buildPythonPackage {
    inherit pname version;
    format = "pyproject";
    src = fetchPypi {
        inherit pname version;
        hash = "sha256-RBloVPQMX5zqMUKCS58QUfha/D/PdZPsVHn8jbAcWNs=";
    };
    build-system = [setuptools];
    dependencies = [numpy scipy opt-einsum ml-dtypes jaxlib] ++ 
      lib.optionals (builtins.hasAttr "jax-cuda12-plugin" dependencies) [ dependencies.jax-cuda12-plugin ];
}