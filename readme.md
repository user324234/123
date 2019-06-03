# Original from https://www.reddit.com/r/MachineLearning/comments/akbc11/p_tag_estimation_for_animestyle_girl_image/
# Prerequisites (you can install these libraries via pip command)
scikit-image  
numpy  
cntk-gpu  
click  
flask (for server)

# Usage
Download model from https://koto.reisen/model.cntk and put in danbooru-resnet_custom_v2-p4/ or run get_model.sh for unix systems  
## For sidecar tagger
> python tagger.py evaluate "danbooru-resnet_custom_v1-p4" "some image.jpg"  
> python tagger.py evaluate-batch "danbooru-resnet_custom_v1-p4" "some folder"  

or see --help option.
## For server
> python server.py

### Lookup script  
![filelookup](DeepDanbooru.png)
