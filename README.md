# SOMACHINE 2020
## Machine Learning, Big Data, and Deep Learning in Astronomy

### A Severo Ochoa School of the Instituto de Astrofísica de Andalucía (CSIC)

![SOMACHINE](tutorials/SOMACHINE_LOGOS.png)


This repository hosts the materials for the school and the conda environment needed to execute them.

## School materials

- **Tutorial 01: Practical ML: Scikit-learn**  (Juan Antonio Cortés, UGR)
    - [Practical ML: Scikit-learn (notebook)](tutorials/tutorial_01_ML/tutorial_01_ML.ipynb)
- **Tutorials 02+03: Big Data: Algorithms and Spark, Data Analysis with Spark** (Diego García, UGR)
    - [Apache Spark installation (pdf)](tutorials/tutorial_02_BD_algorithms_spark/Apache%20Spark%20installation.pdf)
    - VM to be downloaded (see instructions) [link](https://drive.google.com/file/d/1RvOYHH58bNZZ_sbJ_s8xt7gNgkdHzAUr/view?usp=sharing)
- **Tutorial 04: Practical DL with Keras: A Quick Glance** (Alberto Castillo, UGR)
   - [Practical Deep Learning_ A quick glance (pdf)](tutorials/tutorial_04_DL_keras/Practical%20Deep%20Learning_%20A%20quick%20glance.pdf)
   - [Data preparation (notebook)](tutorials/tutorial_04_DL_keras/galaxy_data_preparation.ipynb)
   - [Galaxy Classification (notebook)](tutorials/tutorial_04_DL_keras/galaxy_classification.ipynb)

# Execution of the tutorials

Tutorials 01 and 04 can be followed as Jupyter notebooks using python. Information below shows how to run those notebooks on cloud services or in your local machine. Tutorials 02 and 03 use Spark, which you can install in your machine (see instructions [here](tutorials/tutorial_02_BD_algorithms_spark/Apache%20Spark%20installation.pdf)) or can be executed using the Virtual Machine (VM) provided above.

## Execute notebook tutorials on the cloud

Interactive mybinder link to execute the python notebooks:

[![badge](https://img.shields.io/badge/launch-binder-579ACA.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFkAAABZCAMAAABi1XidAAAB8lBMVEX///9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olL1olJXmsrmZYH1olL1olJXmsrmZYH1olL1olL0nFf1olJXmsrmZYH1olJXmsq8dZb1olJXmsrmZYH1olJXmspXmspXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olJXmsrmZYH1olL1olLeaIVXmsrmZYH1olL1olL1olJXmsrmZYH1olLna31Xmsr1olJXmsr1olJXmsrmZYH1olLqoVr1olJXmsr1olJXmsrmZYH1olL1olKkfaPobXvviGabgadXmsqThKuofKHmZ4Dobnr1olJXmsr1olJXmspXmsr1olJXmsrfZ4TuhWn1olL1olJXmsqBi7X1olJXmspZmslbmMhbmsdemsVfl8ZgmsNim8Jpk8F0m7R4m7F5nLB6jbh7jbiDirOEibOGnKaMhq+PnaCVg6qWg6qegKaff6WhnpKofKGtnomxeZy3noG6dZi+n3vCcpPDcpPGn3bLb4/Mb47UbIrVa4rYoGjdaIbeaIXhoWHmZYHobXvpcHjqdHXreHLroVrsfG/uhGnuh2bwj2Hxk17yl1vzmljzm1j0nlX1olL3AJXWAAAAbXRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hgYGBkcHBwcXl8gICAgoiIkJCQlJicnJ2goKCmqK+wsLC4usDAwMjP0NDQ1NbW3Nzg4ODi5+3v8PDw8/T09PX29vb39/f5+fr7+/z8/Pz9/v7+zczCxgAABC5JREFUeAHN1ul3k0UUBvCb1CTVpmpaitAGSLSpSuKCLWpbTKNJFGlcSMAFF63iUmRccNG6gLbuxkXU66JAUef/9LSpmXnyLr3T5AO/rzl5zj137p136BISy44fKJXuGN/d19PUfYeO67Znqtf2KH33Id1psXoFdW30sPZ1sMvs2D060AHqws4FHeJojLZqnw53cmfvg+XR8mC0OEjuxrXEkX5ydeVJLVIlV0e10PXk5k7dYeHu7Cj1j+49uKg7uLU61tGLw1lq27ugQYlclHC4bgv7VQ+TAyj5Zc/UjsPvs1sd5cWryWObtvWT2EPa4rtnWW3JkpjggEpbOsPr7F7EyNewtpBIslA7p43HCsnwooXTEc3UmPmCNn5lrqTJxy6nRmcavGZVt/3Da2pD5NHvsOHJCrdc1G2r3DITpU7yic7w/7Rxnjc0kt5GC4djiv2Sz3Fb2iEZg41/ddsFDoyuYrIkmFehz0HR2thPgQqMyQYb2OtB0WxsZ3BeG3+wpRb1vzl2UYBog8FfGhttFKjtAclnZYrRo9ryG9uG/FZQU4AEg8ZE9LjGMzTmqKXPLnlWVnIlQQTvxJf8ip7VgjZjyVPrjw1te5otM7RmP7xm+sK2Gv9I8Gi++BRbEkR9EBw8zRUcKxwp73xkaLiqQb+kGduJTNHG72zcW9LoJgqQxpP3/Tj//c3yB0tqzaml05/+orHLksVO+95kX7/7qgJvnjlrfr2Ggsyx0eoy9uPzN5SPd86aXggOsEKW2Prz7du3VID3/tzs/sSRs2w7ovVHKtjrX2pd7ZMlTxAYfBAL9jiDwfLkq55Tm7ifhMlTGPyCAs7RFRhn47JnlcB9RM5T97ASuZXIcVNuUDIndpDbdsfrqsOppeXl5Y+XVKdjFCTh+zGaVuj0d9zy05PPK3QzBamxdwtTCrzyg/2Rvf2EstUjordGwa/kx9mSJLr8mLLtCW8HHGJc2R5hS219IiF6PnTusOqcMl57gm0Z8kanKMAQg0qSyuZfn7zItsbGyO9QlnxY0eCuD1XL2ys/MsrQhltE7Ug0uFOzufJFE2PxBo/YAx8XPPdDwWN0MrDRYIZF0mSMKCNHgaIVFoBbNoLJ7tEQDKxGF0kcLQimojCZopv0OkNOyWCCg9XMVAi7ARJzQdM2QUh0gmBozjc3Skg6dSBRqDGYSUOu66Zg+I2fNZs/M3/f/Grl/XnyF1Gw3VKCez0PN5IUfFLqvgUN4C0qNqYs5YhPL+aVZYDE4IpUk57oSFnJm4FyCqqOE0jhY2SMyLFoo56zyo6becOS5UVDdj7Vih0zp+tcMhwRpBeLyqtIjlJKAIZSbI8SGSF3k0pA3mR5tHuwPFoa7N7reoq2bqCsAk1HqCu5uvI1n6JuRXI+S1Mco54YmYTwcn6Aeic+kssXi8XpXC4V3t7/ADuTNKaQJdScAAAAAElFTkSuQmCC)](https://mybinder.org/v2/gh/spsrc/somachine2020/master?urlpath=lab/tree/tutorials/index.ipynb)

or follow this [link](https://mybinder.org/v2/gh/spsrc/somachine2020/master?urlpath=lab/tree/tutorials/index.ipynb)

[myBinder.org](myBinder.org) is a free and open organization providing free cloud resources. Therefore, the resources may be limited and the changes you make in the notebooks or the system are not persistent. Please, always keep a local copy of any file you want to keep, because Binder will automatically eliminate the virtual machine assigned to you after some time of inactivity.

## Execute notebook tutorials in your local machine

### Install conda

We recommend using `conda` to manage the dependencies. Miniconda is a light-weight version of Anaconda. First we show how to install Miniconda if you don't have it already. More details [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

Miniconda for Linux:
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
rm ./Miniconda3-latest-Linux-x86_64.sh
```

Miniconda for macOS:
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
rm Miniconda3-latest-MacOSX-x86_64.sh
```

Note that the installation will suggest you to modify your bashrc so conda is always available, which is a good idea in general. Alternatively, if you want the Miniconda installation to be encapsulated in your working directory without affecting the rest of your system you can install it with the following option. The first command only needs to be done once, and the second one needs to be done everytime you open a new terminal. 

```bash
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p my_conda_env
source my_conda_env/etc/profile.d/conda.sh
```

### Get the contents of the school

Download this repository and create conda environment with the dependencies
```bash
git clone https://github.com/spsrc/somachine2020.git
cd somachine
conda env create -f environment.yml
conda activate somachine
```

If you want to use Jupyer Lab:
```bash
conda install -c conda-forge jupyterlab
jupyter lab
```
