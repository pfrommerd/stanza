{buildPythonPackage, env, nixpkgs, python, fetchurl} : buildPythonPackage {
    pname = "numpy";
    version = "2.0.1";
    src = fetchurl {
        url="https://files.pythonhosted.org/packages/1c/8a/0db635b225d2aa2984e405dc14bd2b0c324a0c312ea1bc9d283f2b83b038/numpy-2.0.1.tar.gz";
        hash="sha256-SFuHI1eWQQw1GaaZz+H6qwl+UJ6Q67BdzQmNsq6H57M=";
    };
    build-system = [env.meson-python env.cython];
    doCheck = false;
}
