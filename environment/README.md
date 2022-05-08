## 环境安装

---

<br>

### DeepMind Lab 环境

<br>

**2022-05-08 安装**

---

#### 服务器系统，及Anaconda虚拟环境：
   * Ubuntu 20.04
   * python 3.7
   * pytorch 1.10.2
   * mlagent 0.28.0

（列出了安装过程中看到的相关环境，实际安装中好像会自动下载更新调整版本）

<br>

#### 安装步骤

---

**1. 安装依赖环境**

<code>sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3</code>

<br>

**2. 安装 Bazel**

从 Lab 仓库的 README 跳转到的 Bazel 官网写的方法只支持 Ubuntu16.04 和 Ubuntu 18.04，不过最新的 Bazel 版本支持 Ubuntu 20.04。

Ubuntu 20.04 安装 Bazel 参考了大佬的方法 https://gist.github.com/diegopacheco/10f4d7be75574e53e91c49f19bf2613f

其中的第二条命令需要科学上网

<code>sudo apt install curl gnupg</code>

<code>curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg</code>

<code>sudo mv bazel.gpg /etc/apt/trusted.gpg.d/</code>

<code>echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list</code>

<code>sudo apt update && sudo apt install bazel</code>

<code>sudo apt update && sudo apt full-upgrade</code>

<code>bazel --version</code>

能够成功输出 Bazel 版本就说明装好了（2022-05-08 安装的 Bazel 版本是 5.1.1）


<br>

**3. 安装 Python API: deepmind_lab**

要注意官方给的安装说明文档中的 python 版本区别，如果虚拟环境是 python3 的话可以执行一下代码安装 deepmind_lab

<code>cd lab</code>

<code>bazel build -c opt --python_version=PY3 //python/pip_package:build_pip_package</code>

<code>./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg</code>

<code>pip install /tmp/dmlab_pkg/deepmind_lab-1.0-py3-none-any.whl --force-reinstall</code>

不出意外应该是可以顺利执行通过上述命令的，要是有问题那我也没办法，毕竟我也只在自己服务器上测试通过。。。

<br>

#### 测试一下官方给的小例子看是否装好（照搬了）

---

Create a new file `agent.py` and add the following:

```python
import deepmind_lab
import numpy as np

# Create a new environment object.
lab = deepmind_lab.Lab("demos/extra_entities", ['RGB_INTERLEAVED'],
                       {'fps': '30', 'width': '80', 'height': '60'})
lab.reset(seed=1)

# Execute 100 walk-forward steps and sum the returned rewards from each step.
print(sum(
    [lab.step(np.array([0,0,0,1,0,0,0], dtype=np.intc)) for i in range(0, 100)]))
```

Run `agent.py`:

```sh
(agentenv)$ python agent.py
```

DeepMind Lab prints debugging/diagnostic info to the console, but at the end it
should print out a number showing the reward. For the map in the example, the
reward should be 4.0 (since there are four apples in front of the spawn).
