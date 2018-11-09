import sqlite3
import numpy as np; import pandas as pd
import re; import datetime; from collections import Counter
import category_encoders as ce
from sklearn.model_selection import train_test_split
import xgboost as xgb

con = sqlite3.connect(r'C:\Users\myeon\Desktop\Data Science\project\db\race.db')
cursor = con.cursor()

cursor.execute("SELECT total_hn_bu.date, total_hn_bu.no, total_hn_bu.distance, total_hn_bu.horse_name, total_hn_bu.horse_no, \
                total_hn_bu.horse_weight, total_hn_bu.horse_weight_pm, total_hn_bu.location, total_hn_bu.race_time, \
                total_hn_bu.road_cd_pc, total_hn_bu.weather, total_hn_bu.weight_up, total_hn_bu.경기등급1, total_hn_bu.기수명, \
                total_hn_bu.단승, total_hn_bu.번호, total_hn_bu.복승식, total_hn_bu.산지, total_hn_bu.성별, total_hn_bu.순위, \
                total_hn_bu.연령, total_hn_bu.연승, total_hn_bu.중량, grade.등급 FROM total_hn_bu, grade \
                WHERE total_hn_bu.horse_weight <> 0 AND total_hn_bu.race_time <> '' AND total_hn_bu.road_cd_pc IS NOT NULL AND \
                total_hn_bu.weather <> '' AND total_hn_bu.복승식 <> 0 AND grade.horse_no = total_hn_bu.horse_no AND\
                grade.시작일자 <= total_hn_bu.date AND grade.종료일자 >= total_hn_bu.date ORDER BY total_hn_bu.horse_no, \
                total_hn_bu.date")
                # 0 / 5 / 9 / 14 / 20
    
total = pd.DataFrame(cursor.fetchall())

def d_mnth(x, y):
    t = (x//100)%100
    if t <= y: return x - 8800 - y * 100
    else: return x-y*100
    
def rcd_to_sec(x):
    return float(x[0]) * 60 + float(x[2:])

def sex(x):
    if x == '암': return 0
    elif x == '수': return 1
    else: return 2

num = re.compile('[0-9]')
# 뽑을 것 total[1,2,8,9,11,14,15,16,20,21,22], weight, location, rate, race_grade, weather, nation, sex, tmp // 8은 race_record임
 
def rating(x):
    tmp = list(num.findall(x))#; print(tmp)
    if tmp == []: return 10
    else: return int(tmp[0])    
    
### Data Pre-processing -- 매우 지저분함.
    
total[8] = total[8].apply(rcd_to_sec) # 기록을 초 단위 변경
weight = total[5] + total[6] # 최종 중량 도출
location = total[7] == 'seo' # 서울이면 1 아니면 0
rate = total[23].apply(rating) # 말의 등급
race_grade = total[12].apply(rating) # 
    
le =  ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")
weather = le.fit_transform(list(total[10])); nation = le.fit_transform(list(total[17])); sex = total[18].apply(sex)

horses = list(total.iloc[:,4]); cnt = Counter(horses)
dates = list(total.iloc[:,0]); t = 0; tmp = []

for i in cnt.keys():
    cursor.execute("SELECT 진료일자 FROM cure WHERE horse_no = {0}".format(i))
    cure = cursor.fetchall()
    cure = np.array(cure).reshape(len(cure))
    for j in range(cnt[i]):
#       print(dates[t], d_mnth(dates[t],1))
        recent1 = cure[np.where(cure >= d_mnth(dates[t],1))]; recent1 = recent1[np.where(recent1 < dates[t])] # 1개월 이내 진료 여부
#       print(recent1)
        if len(recent1) == 0: tmp.append(0)
        else: tmp.append(1)
        t += 1
#    print(cure); print(dates[:cnt[i]])
#    break

con.close()

### 데이터 최종

df = np.array(total[8]).reshape(-1,1)

idxx = [1,2,9,11,14,15,16,20,21,22]
for i in idxx:
    df = np.append(df, np.array(total[i]).reshape(-1,1), axis = 1)
# 라운드,거리,거리상태,체중증감여부,단승,번호,복승식,연령,연승률,중량,체중,장소,레이팅,경기등급,날씨5개,국적n개,성별,병원
df = np.append(df, np.array(weight).reshape(-1,1), axis = 1)
df = np.append(df, np.array(location.apply(int)).reshape(-1,1), axis = 1)
df = np.append(df, np.array(rate).reshape(-1,1), axis = 1)
df = np.append(df, np.array(race_grade).reshape(-1,1), axis = 1)
df = np.append(df, weather, axis = 1)
df = np.append(df, nation, axis = 1)
df = np.append(df, np.array(sex).reshape(-1,1), axis = 
df = np.append(df, np.array(tmp).reshape(-1,1), axis = 1)
               
### Learning
               
train, test = train_test_split(df, test_size = 0.3, random_state=datetime.datetime.now().second)
real = test[:,0]; real2 = train[:,0]
train = xgb.DMatrix(train[:,1:], label=train[:,0]) # xgb에서 쓸 수 있게 자료형 변경
test = xgb.DMatrix(test[:,1:], label=test[:,0])

# 이제 xgboost 돌리자~
param = {'max_depth':20, 'eta':0.09, 'gamma':0, 'lambda':1, 'silent':1, 'objective':'reg:linear', 'subsample':0.9, 'colsample_bytree':0.8} # parameter 설정: 공부 필요 - linear??? 만약 각 leaf에서 linear reg 추정이라면 one-hot할 게 훨씬 많아짐
num_round = 200

bst = xgb.train(param, train, num_round) # train
preds = bst.predict(test) # test
preds2 = bst.predict(train)

print(abs(preds - real).mean())
print(abs(preds2 - real2).mean())# rmse 출력
