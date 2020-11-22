# High-Value-Customer-Forecast-Competition

## Background
Attended the compettion - I'm the Best Coder! Challenge 2020 - hosted by Shopee Taiwan and won the first place.  
- Official Website: https://careers.shopee.tw/bestcoder/  


## Objective
We were given 6-month purchase details, login information, and user profile, ranging from February to July, to predict high-value users in August 2020. High-value users are defined by whose monthly total GMV (Gross Merchandise Volume) being greater than 70th percentile.
- Kaggle InClass Competition Link: https://www.kaggle.com/c/iamthebestcoderopen2020/overview

## Architecture
### Overview

### Feature Engineering
As to the features, we took RFM model as the reference to create features based on user behaviors which distinguish the high-value users from others. 

- [RFM model](https://en.wikipedia.org/wiki/RFM_(market_research))


### Model Training
The model is basically developed on Lightgbm. After identifying and synthesizing valuable features, the model then predicted the labels based on those features.
- Lightgbm: https://lightgbm.readthedocs.io/en/latest/


### Inferencing
We use ```soft voting``` method to make the final submission.
In other words, we averaged the predicted probabilities for each userid.
Please see ```Voting.ipynb``` for details.



## Reproduction
1. Execute ```mkdir data``` under repo folder
2. Download raw data from the ```Kaggle Inclass competition link``` and put them in ```data``` folder.
3. Run ```shopee.R``` and ```purchase_monthly_pivot.py```. And move the output features into ```data``` folder.
4. Run ```modeling.ipynb```.

## Contributor
- [Tang-Li-Jen](https://github.com/Tang-Li-Jen)
- [Charlie Wang](https://github.com/wwater-wang)
- [kunw-ho](https://github.com/kunw-ho)
- [leo8031](https://github.com/leo8031)

