# High-Value-Customer-Forecast-Competition

## Background
Attended the compettion - I'm the Best Coder! Challenge 2020 - hosted by Shopee Taiwan and won the first place.  
- Official Website: https://careers.shopee.tw/bestcoder/  


## Objective
We were given 6-month purchase details, login information, and user profile, ranging from February to July, to predict high-value users in August 2020. High-value users are defined by whose monthly total GMV (Gross Merchandise Volume) being greater than 70th percentile.
- Kaggle InClass Competition Link: https://www.kaggle.com/c/iamthebestcoderopen2020/overview

## Architecture
### Overview
![ml pipeline](https://github.com/Tang-Li-Jen/High-Value-Customer-Forecast-Competition/blob/main/img/Shopee-Competition-flowchart.jpeg)

### Feature Engineering
As to the features, we took RFM model as the reference to create features based on user behaviors which distinguish the high-value users from others. 

- [RFM model](https://en.wikipedia.org/wiki/RFM_(market_research))

For example, we use recency from last logging or purchase detail as features. Recency has intuitive explanation that the longer time ago the last purchase/logging users have, the lower chances they show up and purchase. 
As for frequency, we divided this dimension into different levels:

1. Distinct days or single counts
2. Given multiple periods (within last 3, 7, 14, 30, 60, 90 days)
3. Avg. growth rate (MoM)
4. Avg. interval among events

We combined these levels to produce multiple features, such as: how many purchases (distinct days) the user has within 30 days? Or we use growth rates and intervals to estimate if this user has intention to purchase more frequently.
When it comes to monetary side, due to the lack of consumption data, we use single counts on purchase ignoring categories. On the other hands, though shopee offers de-identification category instead of clear labels, we still believe some categories have higher prices and contribute great information. Therefore, we count purchases group by category, and then pivot them into multiple features.


### Model Training
The model is basically developed on Lightgbm. After identifying and synthesizing valuable features, the model then predicted the labels based on those features.
- Lightgbm: https://lightgbm.readthedocs.io/en/latest/


### Inferencing
We use ```soft voting``` method to make the final submission.
In other words, we averaged the predicted probabilities from top N submissions by userid.
Please see ```Voting.ipynb``` for details.



## Reproduction
1. Execute ```mkdir data``` under repo folder
2. Download raw data from the ```Kaggle Inclass competition link``` and put them in ```data``` folder.
3. Run ```shopee.R``` and ```purchase_monthly_pivot.py```. And move the output features into ```data``` folder.
4. Run ```modeling.ipynb```.

## Contributors
- [Tang-Li-Jen](https://github.com/Tang-Li-Jen)
- [Charlie Wang](https://github.com/wwater-wang)
- [kunw-ho](https://github.com/kunw-ho)
- [leo8031](https://github.com/leo8031)

