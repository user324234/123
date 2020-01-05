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

# Installation
Download model from https://koto.reisen/model.h5 and put in model/ folder or run get_model.sh for unix systems.  
Run `pip install . --user` or `python setup.py install --user` in folder.  
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

* Suppress tensorflow output by setting the `TF_CPP_MIN_LOG_LEVEL` environment variable  
