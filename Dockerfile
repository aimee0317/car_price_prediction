# Author: Amelia Tang 

FROM jupyter/minimal-notebook

USER root

# R pre-requisites
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    fonts-dejavu \
    unixodbc \
    unixodbc-dev \
    r-cran-rodbc \
    gfortran \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# R packages including IRKernel which gets installed globally.
# r-e1071: dependency of the caret R package
RUN mamba install --quiet --yes \
    'r-base=4.1' \
    'r-caret' \
    'r-crayon' \
    'r-devtools' \
    'r-e1071' \
    'r-hexbin' \
    'r-htmltools' \
    'r-htmlwidgets' \
    'r-irkernel' \
    'r-rcurl' \
    'r-knitr' \
    'r-rodbc' \
    'unixodbc' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# These packages are not easy to install under arm
# hadolint ignore=SC2039
RUN set -x && \
    arch=$(uname -m) && \
    if [ "${arch}" == "x86_64" ]; then \
    mamba install --quiet --yes \
    'r-rmarkdown' \
    'r-tidymodels' \
    'r-tidyverse' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"; \
    fi;


RUN conda install --quiet --y \ 
    _py-xgboost-mutex \
    altair \ 
    altair_data_server \
    altair_saver \
    altair_viewer \ 
    appnope \ 
    asttokens \ 
    async_generator \ 
    atk-1.0 \ 
    attrs \ 
    backcall \ 
    backports \ 
    backports.functools_lru_cache \ 
    brotli \ 
    brotli-bin \ 
    brotlipy \ 
    bzip2 \ 
    ca-certificates \ 
    cairo \ 
    certifi \ 
    cffi \ 
    charset-normalizer \ 
    click \ 
    cloudpickle \
    colorama \ 
    comm \ 
    contourpy \
    cryptography \ 
    cycler \ 
    dataclasses \ 
    debugpy \ 
    decorator \ 
    docopt \ 
    entrypoints \ 
    exceptiongroup \ 
    executing \ 
    expat \ 
    flask \ 
    font-ttf-dejavu-sans-mono \ 
    font-ttf-inconsolata \ 
    font-ttf-source-code-pro \ 
    font-ttf-ubuntu \ 
    fontconfig \ 
    fonts-conda-ecosystem \ 
    fonts-conda-forge \ 
    fonttools \ 
    freetype \ 
    fribidi \ 
    gdk-pixbuf \ 
    gettext \ 
    giflib \ 
    graphite2 \ 
    graphviz \ 
    gtk2 \ 
    gts \ 
    h11 \ 
    harfbuzz \ 
    htmlmin \ 
    icu \ 
    idna \ 
    imagehash \ 
    importlib-metadata \ 
    importlib_resources \ 
    ipykernel \ 
    ipython \ 
    ipywidgets \ 
    itsdangerous \ 
    jedi \ 
    jinja2 \ 
    joblib \ 
    jpeg \ 
    jsonschema \ 
    jupyter_client \
    jupyter_core \ 
    jupyterlab_widgets \ 
    kiwisolver \ 
    lcms2 \ 
    lerc \ 
    libblas \ 
    libbrotlicommon \ 
    libbrotlidec \ 
    libcblas \ 
    libcxx \ 
    libdeflate \ 
    libffi \ 
    libgd \ 
    libgfortran5 \ 
    libglib \ 
    libiconv \ 
    libjpeg-turbo \ 
    liblapack \ 
    libllvm11 \ 
    libopenblas \ 
    libpng \ 
    librsvg \ 
    libsodium \ 
    libsqlite \ 
    libtool \ 
    libuv \ 
    libwebp \ 
    libxcb \ 
    libxml2 \ 
    libzlib \ 
    llvm-openmp \ 
    llvmlite \ 
    markupsafe \ 
    matplotlib \ 
    matplotlib-base \ 
    matplotlib-inline \ 
    multimethod \ 
    munkres \ 
    nbformat \ 
    ncurses \ 
    nest-asyncio \ 
    networkx \ 
    numba \ 
    numpy \ 
    openjpeg \ 
    openssl \ 
    outcome \ 
    packaging \ 
    pandas \ 
    pandas-profiling \ 
    pango \ 
    patsy \ 
    parso \
    pcre2 \ 
    pexpect \ 
    phik \ 
    pickleshare \ 
    pillow \ 
    pip \ 
    pixman \ 
    pkgutil-resolve-name \ 
    platformdirs \ 
    portpicker \ 
    prompt-toolkit \ 
    psutil \ 
    pthread-stubs \ 
    ptyprocess \ 
    pure_eval \ 
    py-xgboost \ 
    pybind11-abi \ 
    pycparser \ 
    pygments \ 
    pyopenssl \ 
    pyparsing \ 
    pyrsistent \ 
    pysocks \ 
    python \ 
    python-dateutil \ 
    python-fastjsonschema \ 
    python-graphviz \ 
    python_abi \ 
    pytz \ 
    pywavelets \ 
    pyyaml \ 
    pyzmq \ 
    readline \ 
    requests \ 
    scikit-learn \ 
    scipy \ 
    seaborn-base \ 
    selenium \ 
    setuptools \ 
    shap \ 
    six \ 
    slicer \ 
    sniffio \ 
    sortedcontainers \ 
    stack_data \ 
    statsmodels \ 
    tangled-up-in-unicode \ 
    threadpoolctl \ 
    tk \ 
    toolz \ 
    tornado \ 
    tqdm \ 
    traitlets \ 
    trio \ 
    trio-websocket \ 
    typeguard \ 
    typing-extensions \ 
    tzdata \ 
    unicodedata2 \ 
    urllib3 \ 
    visions \ 
    wcwidth \ 
    werkzeug \ 
    wheel \ 
    widgetsnbextension \ 
    wsproto \ 
    xgboost \ 
    xorg-libxau \ 
    xorg-libxdmcp \ 
    xz \ 
    yaml \ 
    zeromq \ 
    zipp \ 
    zlib \ 
    zstd 
    
# Install dependancy for altair to save png files
RUN npm install -g npm vega vega-cli vega-lite canvas -f


# Install R table graphics for final report
RUN conda install r-kableextra
