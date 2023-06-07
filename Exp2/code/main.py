import os,sys
import urllib.request as ur
import urllib.error as ue
import json,csv
from time import sleep
from random import random

from web_utils import retrieve_info,get_course_list

keys=["序号","课程名称","课程难度","作业多少","给分好坏","收获大小",\
      "选课类别","教学类型","课程类型","开课单位","课程层次","学分"]

def save_info(data_list:list[dict],csv_path="./icourse.csv",json_path="./icourse.json")->None:
  '''save the course info as both a json file and a csv file'''
  os.chdir(sys.path[0]) #use relevant file address

  with open(json_path,mode="w",encoding='utf8') as f_out:
    json.dump(data_list,f_out,indent=2,ensure_ascii=False)
    f_out.close()

  csv_list:list[list[str]]=list()
  for info in data_list:
    csv_list.append(list(info.values()))

  with open(csv_path,encoding="utf-8",mode="w") as f_out:
    csvWriter=csv.writer(f_out)
    csvWriter.writerow(keys)
    csvWriter.writerows(csv_list)
    f_out.close()    

def open_web(url:str)->str | None:
  '''open target web and return web source as string\n
  if failed, return None'''
  try:
    #some murmur from the ancient
    req=ur.Request(url)
    req.add_header('User-Agent',"Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                  AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 \
                  Safari/537.36 Edge/17.17134")  
    req.add_header('Host','httpbin.org')
    IP=ur.ProxyHandler({'http':"218.22.65.606:0088"})
    opener=ur.build_opener(IP,ur.HTTPHandler)
    response=opener.open(url)

    web=str(response.read().decode('utf-8'))
    return web
  
  except ue.URLError as e:
    print(e.reason) #Not Found
    return None

def main()->bool:
  record_need:int=200  #record needed
  record_count:int=0  #current count of record
  page_count:int=1  #the course list page number
  page_count_max:int=50 #limit of pages of course list
  data_list:list[dict[str,str]]=list() #course info list

  #sequentially search the main listing page till meet the demand count
  while(record_count<record_need and page_count<=page_count_max):
    url_course_list_page="https://icourse.club/course/?page="+str(page_count)
    page_count+=1

    #open the page of listing out 10 course summuries
    new_web=open_web(url_course_list_page)
    url_course_list=get_course_list(new_web)    
    if url_course_list==None:
      continue
    
    #open these 10 pages seperately for individual course info
    for url in url_course_list:
      new_web=open_web(url)
      
      new_course_info=retrieve_info(new_web,record_count)
      if new_course_info==None:
        continue
      else:
        data_list.append(new_course_info)
        record_count+=1

    #considering the risk of anti-crawler
    sleep(random()*1.0+1.0)  
    print("record count:"+str(record_count)+"\n")
  
  #save course info we got(maybe less than target demand)
  save_info(data_list)
  if record_count<record_need:
    return False
  else:
    return True


if __name__ =="__main__":
  main()
