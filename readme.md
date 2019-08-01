# Original from https://www.reddit.com/r/MachineLearning/comments/akbc11/p_tag_estimation_for_animestyle_girl_image/
# Dependencies
python 3.6  
scikit-image  
numpy  
cntk-gpu  
click  
flask (for server)  
[Hydrus API](https://gitlab.com/cryzed/hydrus-api)(for API intergration)

# Usage
Download model from https://koto.reisen/model.cntk and put in danbooru-resnet_custom_v2-p4/ or run get_model.sh for unix systems  
## For sidecar tagger
> python tagger.py evaluate "danbooru-resnet_custom_v1-p4" "some image.jpg"  

> python tagger.py evaluate-batch "danbooru-resnet_custom_v1-p4" "some folder"  

or see `--help` option.
## For server
> python server.py

## For API intergration
> python tagger.py evaluate-api-hash "danbooru-resnet_custom_v1-p4" --hash 52f7ab1c5860ef3d9b71d0fc3e69676fb2c2da16deaa8cb474ef20043ef43f30 --api_key 466f2185417001876effabd9ab53f9447439958b0774bf50d262b109e598ee99  

> python tagger.py evaluate-api-hash "danbooru-resnet_custom_v1-p4" --input hashes.txt --api_key 466f2185417001876effabd9ab53f9447439958b0774bf50d262b109e598ee99  

> python tagger.py evaluate-api-search "danbooru-resnet_custom_v1-p4" --api_key 466f2185417001876effabd9ab53f9447439958b0774bf50d262b109e598ee99 "1girl" "brown hair" "blue eyes"  

or see `python tagger.py evaluate-api --help`  

You can add a default `api_key` into `tagger.py` by adding it to the DEFAULT_API_KEY variable.  
```DEFAULT_API_KEY = "466f2185417001876effabd9ab53f9447439958b0774bf50d262b109e598ee99"```  
like so.  

### Lookup script  
![filelookup](DeepDanbooru.png)
