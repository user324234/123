# DeepDanbooru Model from https://github.com/KichangKim/DeepDanbooru/
# Dependencies
click  
flask (for server)  
hydrus-api (for API intergration)  
tensorflow>=2  
scikit-image  
numpy  
six  
appdirs  
Pillow  

# Installation
Download model from https://koto.reisen/model.h5 or through deepdanbooru github releases and put in model/ folder.  
You can run ./get_model.sh on unix systems to automatically download latest model into the model folder.  
You can also download the older v1, older v3, or v4 models from https://koto.reisen/model_v1.h5, https://koto.reisen/model_v3.h5 or https://koto.reisen/model_v4.h5, be sure to use with tags_v1.txt, tags_v3 or tags_v4.txt respectively.  
Run `pip install . --user` or `python setup.py install --user` in folder.  
For poetry installation run `poetry install` in folder.  
Some dependencies are optional so you will have to use the extras flag in pip or poetry to install them.  
Example:  
`pip install '.[server,tensorflow]' --user`  
`poetry install -E server -E api`  
# Configuration
See the [Configuration Page](https://gitgud.io/koto/hydrus-dd/-/wikis/Configuration)
# Usage
## For sidecar hydrus-dd
> hydrus-dd evaluate "some image.jpg"  

> hydrus-dd evaluate-batch "some folder"  

or see `--help` option.
## For server
> hydrus-dd run-server  

see `hydrus-dd run-server --help` for more options  

## For API intergration
> hydrus-dd evaluate-api-hash --hash 52f7ab1c5860ef3d9b71d0fc3e69676fb2c2da16deaa8cb474ef20043ef43f30 --api_key 466f2185417001876effabd9ab53f9447439958b0774bf50d262b109e598ee99  

> hydrus-dd evaluate-api-hash --input hashes.txt --api_key 466f2185417001876effabd9ab53f9447439958b0774bf50d262b109e598ee99  

> hydrus-dd evaluate-api-search --api_key 466f2185417001876effabd9ab53f9447439958b0774bf50d262b109e598ee99 "1girl" "brown hair" "blue eyes"  

or see `hydrus-dd evaluate-api-hash/evaluate-api-search --help`  

### Lookup script  
![filelookup](DeepDanbooru.png)


## Troubleshooting  

* If you are having problems connecting to the filelookup server on Windows on 0.0.0.0, connect to your local IP address instead, and change the IP to it in the filelookup script.  

* Suppress tensorflow output by setting the `TF_CPP_MIN_LOG_LEVEL` environment variable  