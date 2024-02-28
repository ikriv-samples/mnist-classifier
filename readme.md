Simple Image Classifier Based On PyTorch
==========
This sample implements a web site that allows user to draw a digit in a canvas and then submit it for recognition to a PyTorch-based model. The model is based on this repo: https://github.com/GokuMohandas/oreilly-pytorch/tree/master/code.

# Local Setup
On Windows, run `setup.bat`.

# Training
`python3 api/mnist/train.py`
This creates file `api/mnist/nist_model.pt`

# Using Test Set
`python3 api/mnist/test.py`

# Setting up Apache
This repo uses WSGI interface.
Apache should have mod_wsgi installed. Dockerfile fragment:
```
RUN apt install -y libapache2-mod-wsgi-py3
RUN apt install -y curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o/tmp/get-pip.py
RUN python3 /tmp/get-pip.py

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install matplotlib
RUN pip3 install Pillow
```
# Deploying Code
The following should be added to web site configuration in Apache configs:

```
WSGIScriptAlias /api/wsgi $WEBSITE_ROOT/api/wsgi.py
WSGIDaemonProcess website.domain.com processes=2 threads=15 display-name=%{GROUP}
WSGIApplicationGroup %{GLOBAL}
```

`html` folder goes to `$WEBSITE_ROOT/html`
`api`  folder goes to `$WEBSITE_ROOT/api`
