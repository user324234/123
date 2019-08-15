# Original from https://www.reddit.com/r/MachineLearning/comments/akbc11/p_tag_estimation_for_animestyle_girl_image/
# Dependencies
python 3.6  
scikit-image  
numpy  
cntk-gpu  
click  
flask (for server)  
hydrus-api (for API intergration)  

# Usage
Download model from https://koto.reisen/model.cntk and put in danbooru-resnet_custom_v2-p4/ or run get_model.sh for unix systems  
## For sidecar hydrus-dd
> python hydrus-dd.py evaluate "danbooru-resnet_custom_v1-p4" "some image.jpg"  

> python hydrus-dd.py evaluate-batch "danbooru-resnet_custom_v1-p4" "some folder"  

or see `--help` option.
## For server
> python hydrus-dd.py run-server danbooru-resnet_custom_v1-p4"  

see `python hydrus-dd.py run-server --help` for more options  

## For API intergration
> python hydrus-dd.py evaluate-api-hash "danbooru-resnet_custom_v1-p4" --hash 52f7ab1c5860ef3d9b71d0fc3e69676fb2c2da16deaa8cb474ef20043ef43f30 --api_key 466f2185417001876effabd9ab53f9447439958b0774bf50d262b109e598ee99  

> python hydrus-dd.py evaluate-api-hash "danbooru-resnet_custom_v1-p4" --input hashes.txt --api_key 466f2185417001876effabd9ab53f9447439958b0774bf50d262b109e598ee99  

> python hydrus-dd.py evaluate-api-search "danbooru-resnet_custom_v1-p4" --api_key 466f2185417001876effabd9ab53f9447439958b0774bf50d262b109e598ee99 "1girl" "brown hair" "blue eyes"  

or see `python hydrus-dd.py evaluate-api --help`  

You can add a default `api_key` into `hydrus-dd.py` by adding it to the DEFAULT_API_KEY variable.  
```DEFAULT_API_KEY = "466f2185417001876effabd9ab53f9447439958b0774bf50d262b109e598ee99"```  
like so.  

### Lookup script  
![filelookup](DeepDanbooru.png)
