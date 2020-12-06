library(tidyverse)
library(lubridate)

login <- read_csv('../data/login.csv')
purch_detail <- read_csv('../data/purchase_detail.csv')
submission <- read_csv('../data/submission.csv')
user_info <- read_csv('../data/user_info.csv')
user_label_train <- read_csv('../data/user_label_train.csv')

onlyId <- user_info %>%
  select(userid)

#Recency
abt <- login %>%
  group_by(userid) %>%
  summarise(max_dt = max(date)) %>%
  mutate(rececny = ymd(20200731) - max_dt) %>%
  select(-max_dt) %>%
  right_join(user_info, by = c("userid" = "userid"))

abt

#### freq ####
#freq in 3 days
abt <- login %>%
  filter(date >= ymd(20200731)-3) %>%
  group_by(userid) %>%
  summarise(FreqIn3days = sum(login_times)) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(FreqIn3days = ifelse(is.na(FreqIn3days), 0L, FreqIn3days))

#freq in 7 days
abt <- login %>%
  filter(date >= ymd(20200731)-7) %>%
  group_by(userid) %>%
  summarise(FreqIn7days = sum(login_times)) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(FreqIn7days = ifelse(is.na(FreqIn7days), 0L, FreqIn7days))

#freq in 14 days
abt <- login %>%
  filter(date >= ymd(20200731)-14) %>%
  group_by(userid) %>%
  summarise(FreqIn14days = sum(login_times)) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(FreqIn14days = ifelse(is.na(FreqIn14days), 0L, FreqIn14days))

#freq in 30 days
abt <- login %>%
  filter(date >= ymd(20200731)-30) %>%
  group_by(userid) %>%
  summarise(FreqIn30days = sum(login_times)) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(FreqIn30days = ifelse(is.na(FreqIn30days), 0L, FreqIn30days))

#freq in 60 days
abt <- login %>%
  filter(date >= ymd(20200731)-60) %>%
  group_by(userid) %>%
  summarise(FreqIn60days = sum(login_times)) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(FreqIn60days = ifelse(is.na(FreqIn60days), 0L, FreqIn60days))

#freq in 90 days
abt <- login %>%
  filter(date >= ymd(20200731)-90) %>%
  group_by(userid) %>%
  summarise(FreqIn90days = sum(login_times)) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(FreqIn90days = ifelse(is.na(FreqIn90days), 0L, FreqIn90days))

#freq
abt <- login %>%
  group_by(userid) %>%
  summarise(Freq = sum(login_times)) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(Freq = ifelse(is.na(Freq), 0L, Freq))

## freq distinct days
# 3 distinct days
abt <- login %>%
  filter(date >= ymd(20200731)-3) %>%
  group_by(userid) %>%
  summarise(DistinctDayIn3days = n()) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(DistinctDayIn3days = ifelse(is.na(DistinctDayIn3days), 0L, DistinctDayIn3days))

# 7 distinct days
abt <- login %>%
  filter(date >= ymd(20200731)-7) %>%
  group_by(userid) %>%
  summarise(DistinctDayIn7days = n()) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(DistinctDayIn7days = ifelse(is.na(DistinctDayIn7days), 0L, DistinctDayIn7days))

# 14 distinct days
abt <- login %>%
  filter(date >= ymd(20200731)-14) %>%
  group_by(userid) %>%
  summarise(DistinctDayIn14days = n()) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(DistinctDayIn14days = ifelse(is.na(DistinctDayIn14days), 0L, DistinctDayIn14days))

# 30 distinct days
abt <- login %>%
  filter(date >= ymd(20200731)-30) %>%
  group_by(userid) %>%
  summarise(DistinctDayIn30days = n()) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(DistinctDayIn30days = ifelse(is.na(DistinctDayIn30days), 0L, DistinctDayIn30days))

# 60 distinct days
abt <- login %>%
  filter(date >= ymd(20200731)-60) %>%
  group_by(userid) %>%
  summarise(DistinctDayIn60days = n()) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(DistinctDayIn60days = ifelse(is.na(DistinctDayIn60days), 0L, DistinctDayIn60days))

# 90 distinct days
abt <- login %>%
  filter(date >= ymd(20200731)-90) %>%
  group_by(userid) %>%
  summarise(DistinctDayIn90days = n()) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(DistinctDayIn90days = ifelse(is.na(DistinctDayIn90days), 0L, DistinctDayIn90days))

# distinct days
abt <- login %>%
  group_by(userid) %>%
  summarise(DistinctDay = n()) %>%
  right_join(abt, by = c("userid" = "userid")) %>%
  mutate(DistinctDay = ifelse(is.na(DistinctDay), 0L, DistinctDay))

##################
abt2 <- login %>%
  mutate(YYMM = year(date)*100 + month(date)) %>%
  group_by(userid, YYMM) %>%
  summarise(LoginTimes = sum(login_times)) %>%
  arrange(userid, YYMM) %>%
  group_by(lastTimes = lag(LoginTimes)) %>%
  mutate(FreqMoM = (LoginTimes - lastTimes)/lastTimes) %>%
  group_by(userid) %>%
  summarise(avgFreqMoM = mean(FreqMoM, na.rm = TRUE)) %>%
  right_join(onlyId, by = c("userid" = "userid"))

abt2 %>% write_csv("feature2.csv")

#################
### purchase ####
#Recency
abt3 <- purch_detail %>%
  group_by(userid) %>%
  summarise(max_dt = max(grass_date)) %>%
  mutate(BuyRececny = ymd(20200731) - max_dt) %>%
  select(-max_dt) %>%
  right_join(onlyId, by = c("userid" = "userid"))

# Freq
# 3 days
abt3 <- purch_detail %>%
  filter(grass_date >= ymd(20200731) - 3) %>%
  group_by(userid) %>%
  summarise(OrderCntIn3days = sum(order_count),
            TotCntIn3days = sum(total_amount)) %>%
  right_join(abt3, by = c('userid' = 'userid')) %>%
  mutate(OrderCntIn3days = ifelse(is.na(OrderCntIn3days), 0L, OrderCntIn3days),
         TotCntIn3days = ifelse(is.na(TotCntIn3days), 0L, TotCntIn3days))

# 7 days
abt3 <- purch_detail %>%
  filter(grass_date >= ymd(20200731) - 7) %>%
  group_by(userid) %>%
  summarise(OrderCntIn7days = sum(order_count),
            TotCntIn7days = sum(total_amount)) %>%
  right_join(abt3, by = c('userid' = 'userid')) %>%
  mutate(OrderCntIn7days = ifelse(is.na(OrderCntIn7days), 0L, OrderCntIn7days),
         TotCntIn7days = ifelse(is.na(TotCntIn7days), 0L, TotCntIn7days))

# 14 days
abt3 <- purch_detail %>%
  filter(grass_date >= ymd(20200731) - 14) %>%
  group_by(userid) %>%
  summarise(OrderCntIn14days = sum(order_count),
            TotCntIn14days = sum(total_amount)) %>%
  right_join(abt3, by = c('userid' = 'userid')) %>%
  mutate(OrderCntIn14days = ifelse(is.na(OrderCntIn14days), 0L, OrderCntIn14days),
         TotCntIn14days = ifelse(is.na(TotCntIn14days), 0L, TotCntIn14days))

# 30 days
abt3 <- purch_detail %>%
  filter(grass_date >= ymd(20200731) - 30) %>%
  group_by(userid) %>%
  summarise(OrderCntIn30days = sum(order_count),
            TotCntIn30days = sum(total_amount)) %>%
  right_join(abt3, by = c('userid' = 'userid')) %>%
  mutate(OrderCntIn30days = ifelse(is.na(OrderCntIn30days), 0L, OrderCntIn30days),
         TotCntIn30days = ifelse(is.na(TotCntIn30days), 0L, TotCntIn30days))

# 60 days
abt3 <- purch_detail %>%
  filter(grass_date >= ymd(20200731) - 60) %>%
  group_by(userid) %>%
  summarise(OrderCntIn60days = sum(order_count),
            TotCntIn60days = sum(total_amount)) %>%
  right_join(abt3, by = c('userid' = 'userid')) %>%
  mutate(OrderCntIn60days = ifelse(is.na(OrderCntIn60days), 0L, OrderCntIn60days),
         TotCntIn60days = ifelse(is.na(TotCntIn60days), 0L, TotCntIn60days))

# 90 days
abt3 <- purch_detail %>%
  filter(grass_date >= ymd(20200731) - 90) %>%
  group_by(userid) %>%
  summarise(OrderCntIn90days = sum(order_count),
            TotCntIn90days = sum(total_amount)) %>%
  right_join(abt3, by = c('userid' = 'userid')) %>%
  mutate(OrderCntIn90days = ifelse(is.na(OrderCntIn90days), 0L, OrderCntIn90days),
         TotCntIn90days = ifelse(is.na(TotCntIn90days), 0L, TotCntIn90days))

#MoM growth rate on purchase
abt3 <- purch_detail %>%
  mutate(YYMM = year(grass_date)*100 + month(grass_date)) %>%
  group_by(userid, YYMM) %>%
  summarise(OrderCnt = sum(order_count),
            TotCnt = sum(total_amount)) %>%
  arrange(userid, YYMM) %>%
  mutate(LastOrderCnt = lag(OrderCnt),
         LastTotCnt = lag(TotCnt),
         MoMOrderCnt = (OrderCnt - LastOrderCnt)/LastOrderCnt,
         MoMTotCnt = (TotCnt - LastTotCnt)/LastTotCnt) %>%
  group_by(userid) %>%
  summarise(AvgMoMOrderCnt = mean(MoMOrderCnt, na.rm = TRUE),
            AvgMoMTotCnt = mean(MoMTotCnt, na.rm = TRUE)) %>%
  right_join(abt3, by = c('userid' = 'userid'))
  
abt3 %>% write_csv('feature_v3.csv')

###########
abt4 <- purch_detail %>%
  distinct(userid, category_encoded) %>%
  group_by(userid) %>%
  summarise(DistinctCategory = n()) %>%
  right_join(onlyId, by = c('userid' = 'userid')) %>%
  mutate(DistinctCategory = ifelse(is.na(DistinctCategory), 0L, DistinctCategory))

abt4 %>% write_csv('feature_v4.csv')

###########
cate_mapper <- purch_detail %>% 
  group_by(userid, category_encoded) %>%
  summarise(OrderCnt = sum(order_count),
            TotAmt = sum(total_amount)) %>%
  group_by(category_encoded) %>%
  summarise(MedOrderCnt = median(OrderCnt),
            MedTotAmt = median(TotAmt))

abt5 <- purch_detail %>%
  group_by(userid, category_encoded) %>%
  summarise(OrderCnt = sum(order_count),
            TotAmt = sum(total_amount)) %>%
  left_join(cate_mapper, by = c('category_encoded' = 'category_encoded')) %>%
  filter(OrderCnt > MedOrderCnt | TotAmt > MedTotAmt) %>%
  group_by(userid) %>%
  summarise(GoodBuyer = n()) %>%
  right_join(onlyId, by = c('userid' = 'userid')) %>%
  mutate(GoodBuyer = ifelse(is.na(GoodBuyer), 0L, GoodBuyer))

abt5 %>% write_csv('feature_v5.csv')