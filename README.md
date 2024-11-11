# hello-milvus

https://milvus.io/docs/install_standalone-docker-compose.md

### milvus install

```
❯ wget https://github.com/milvus-io/milvus/releases/download/v2.4.15/milvus-standalone-docker-compose.yml -O docker-compose.yml
...
저장 위치: `docker-compose.yml'

❯ docker compose up -d
WARN[0000] /Users/raiz/labs/workspace/hello-milvus/docker-compose.yml: `version` is obsolete
[+] Running 21/3
 ✔ minio Pulled                                                                                                                                                                            8.2s
 ✔ standalone Pulled                                                                                                                                                                      21.7s
 ✔ etcd Pulled                                                                                                                                                                            15.9s
[+] Running 4/4
 ✔ Network milvus               Created                                                                                                                                                    0.1s
 ✔ Container milvus-etcd        Started                                                                                                                                                    0.6s
 ✔ Container milvus-minio       Started                                                                                                                                                    0.6s
 ✔ Container milvus-standalone  Started

❯ docker compose ps
WARN[0000] /Users/raiz/labs/workspace/hello-milvus/docker-compose.yml: `version` is obsolete
NAME                IMAGE                                      COMMAND                   SERVICE      CREATED          STATUS                    PORTS
milvus-etcd         quay.io/coreos/etcd:v3.5.5                 "etcd -advertise-cli…"   etcd         42 seconds ago   Up 41 seconds (healthy)   2379-2380/tcp
milvus-minio        minio/minio:RELEASE.2023-03-20T20-16-18Z   "/usr/bin/docker-ent…"   minio        42 seconds ago   Up 41 seconds (healthy)   0.0.0.0:9000-9001->9000-9001/tcp
milvus-standalone   milvusdb/milvus:v2.4.15                    "/tini -- milvus run…"   standalone   42 seconds ago   Up 40 seconds

❯ docker port milvus-standalone 19530/tcp
0.0.0.0:19530
```

### python install

https://github.com/pyenv/pyenv

```
❯ brew install pyenv
❯ pyenv install 3.10.0
❯ pyenv versions
* system (set by /Users/raiz/.pyenv/version)
  3.10.0
❯ pyenv global 3.10.0
❯ pyenv versions
  system
* 3.10.0 (set by /Users/raiz/.pyenv/version)

❯ brew install pipenv
❯ brew upgrade pipenv
Warning: pipenv 2024.4.0 already installed

❯ pipenv --python 3.10.0
Creating a virtualenv for this project
Pipfile: /Users/raiz/labs/workspace/hello-milvus/Pipfile
Using /Users/raiz/.pyenv/versions/3.10.0/bin/python33.10.0 to create virtualenv...
⠴ Creating virtual environment...created virtual environment CPython3.10.0.final.0-64 in 405ms
  creator CPython3Posix(dest=/Users/raiz/.local/share/virtualenvs/hello-milvus-kq6S-lQc, clear=False, no_vcs_ignore=False, global=False)
  seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/Users/raiz/Library/Application Support/virtualenv)
    added seed packages: pip==24.3.1, setuptools==75.2.0, wheel==0.44.0
  activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator

⠦ Creating virtual environment...✔ Successfully created virtual environment!
Virtualenv location: /Users/raiz/.local/share/virtualenvs/hello-milvus-kq6S-lQc
Creating a Pipfile for this project...

```

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

```

❯ pipenv install pymilvus
Pipfile.lock not found, creating...
Locking [packages] dependencies...
Locking [dev-packages] dependencies...
Updated Pipfile.lock (4c696dfd99024a0c7fcf4ff5a770d9f2b679248f333b938c9d423e64b7929195)!
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
Installing pymilvus...
✔ Installation Succeeded
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
Installing dependencies from Pipfile.lock (929195)...
All dependencies are now up-to-date!
Upgrading pymilvus in  dependencies.
Building requirements...
Resolving dependencies...
✔ Success!
Building requirements...
Resolving dependencies...
✔ Success!
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
Installing dependencies from Pipfile.lock (4fce3e)...
All dependencies are now up-to-date!
Installing dependencies from Pipfile.lock (4fce3e)...
```

### sample

```
❯ wget https://raw.githubusercontent.com/milvus-io/pymilvus/master/examples/hello_milvus.py
...
hello_milvus.py                                 100%[=======================================================================================================>]   7.34K  --.-KB/s    /  0s

2024-11-08 14:09:07 (28.5 MB/s) - `hello_milvus.py' 저장함 [7520/7520]

❯ pip install pymilvus
❯ python hello_milvus.py
```
