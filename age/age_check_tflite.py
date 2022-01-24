import cv2
import numpy as np
import glob
from age_model import AgeResNet
import pandas as pd
import time


class AgeGender:
        def __init__(self, age_model_path="/home/local/ASUAD/sbiookag/guise/age/age/model_current_05_03_2021.tflite"):#/home/gogetter/workspace-prapti/tvm_exp/tflite_tvm/age_gender/models/model_current_05_03_2021.tflite"):
                self.__age_model = AgeResNet(age_model_path)

        @staticmethod
        def get_true_age(logits, probas):
                # print(probas)
                predict_levels = probas > 0.5
                pred_list = probas.tolist()[0]
                max_prob = max(pred_list)
                predicted_label = np.sum(predict_levels, axis=1)

                true_age = str((predicted_label.item()*10)) + ' - ' + \
                            str(((predicted_label.item()+1)*10)-1)
                if str(predicted_label.item()) == '7':
                        true_age = '70+'
                # print (predicted_label.item())
                return true_age, max_prob

        def get_age_gender(self,faces):
        # print ("got faces:",len(faces))
                # detect on frame
                ages={}
                total_time=0
                for file_path in faces:
                        face=cv2.imread(file_path)
                        face= cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

                        img = cv2.resize(face, (120, 120), interpolation = cv2.INTER_AREA)
                        img = np.expand_dims(img.astype(np.float32), 0)#change
                        img = img/255
                        # img = img*256.41-128

                        time_s=time.time()
                        logits, probas = self.__age_model(img)
                        time_end=time.time()-time_s
                        total_time+=time_end

                        age,max_prob= self.get_true_age(logits,probas)

                        ages[file_path]=age
                        # print ("age:",age)


                print(total_time/len(faces))
                return ages  

age_model=AgeGender()
faces=set(glob.glob('/home/local/ASUAD/sbiookag/guise/age/age dataset 02-03-2021/Age_Dataset_02_03_2021_comb/test__*.jpg')[0:50])
ages=age_model.get_age_gender(faces)
age_df=pd.DataFrame()
age_df['path']=list(ages.keys())
age_df['predicted']=list(ages.values())
age_df.to_csv('output_test.csv',index=False)
