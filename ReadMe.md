# Self driving car

Learning deep Q-learning with pytorch (from https://www.udemy.com/artificial-intelligence-az)

<br/>

## Installation of dependencies in a Conda virtualenv

Using Python 3.6.2 on Windows 10:

<br/>
Use the provided conda environment file:

> \> conda env create -f AiCar.yml

<br/>
Alternatively you can install the dependencies manually:

> \> conda install pip  
> \> pip install --upgrade pip setuptools wheel  
> \> pip install docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew  
> \> pip install kivy  
> \> conda install -c peterjc123 pytorch  
> \> conda install matplotlib

<br/>

## Run

> \> python main.py

The car will try to learn how to do round trips frop the bottom-left corner to the top right corner while avoiding the sand you can draw on the map.
