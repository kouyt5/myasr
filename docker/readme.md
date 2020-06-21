# 集群训练
由于数据集较大，在单个计算机上训练花费时间较长，因此准备搭建分布式训练处理流程
## 预准备
很容易想到的是集群为了一致性，对环境肯定有一定要求，下面将会简单介绍一下环境依赖以及搭建步骤
### docker环境安装与配置
为了不破坏主机环境以及保证环境的一致性，因此使用docker解决方案。安装的软件如下：
+ docker 19.0.3
+ docker-compose
+ docker-container-runtime
+ nvidia驱动版本440
+ ubuntu > 16.04(windows 不支持)

为了支持docker-compose启动，需要修改docker一个的配置文件
```shell
$ cat /etc/docker/daemon.json
{
  "registry-mirrors": ["https://vr878luf.mirror.aliyuncs.com"],
  "data-root": "/mnt/volume/docker",
  "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         }
    }
}
```
### 网络配置
为了每个节点能够找到互相的ip地址，需要修改每一台计算机集群的host文件，host文件中注册每个节点的ip和其名字，例如
```
$ cat /etc/hosts
192.168.1.135   yewenzheng-System-Product-Name
192.168.1.104   chencong-research
192.168.1.110   aoneasr-System-Product-Name
192.168.1.102   lisen
```

## 启动步骤

进入docker目录
1. `docker-build -t myasr:v1.1_dist` 构建镜像
2. `docker-compose up` 启动ecid实例，作为集群的管理者，应该是所有的节点都会向改容器发送数据
3. 在每一个结点上 `docker-compose -f docker-compose-train.yaml` 启动每个节点，节点数目在docker-compose-train.yaml的`--nnodes`设置

