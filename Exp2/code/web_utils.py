import re

keys=["序号","课程名称","课程难度","作业多少","给分好坏","收获大小",\
      "选课类别","教学类型","课程类型","开课单位","课程层次","学分"]

def retrieve_info(web:str|None,order:int=0)->dict[str,str]|None:
  '''get the kernel infomation of the course from its detailed review page,\n
  and return a dict of values with preset keys'''
  if web==None:
    return None   
  else:
    web=web.replace("\n"," ") #seems cannot handle a multi-line string text

  web_info_dict:dict[str,str]=dict()

  web_info_dict[keys[0]]=str(order)   #should be int but the example is str
  
  try:
    #curriculum name
    text_name:str=re.findall(r"<title>(.*?)</title>",web)[0]
    text_name=text_name.replace(" - USTC评课社区","") #if there is a suffix
    web_info_dict[keys[1]]=text_name
    '<ul class="text-muted list-inline list-unstyled ud-pd-sm">'
    text_summury:str=re.findall(\
      r'<ul class="text-muted list-inline list-unstyled ud-pd-sm">(.*?)</ul>',web)[0]
    #difficulty
    text_value:str=re.findall(r"课程难度：(.*?)</li>",text_summury)[0]
    web_info_dict[keys[2]]=text_value
    #homework amount
    text_value:str=re.findall(r"作业多少：(.*?)</li>",text_summury)[0]
    web_info_dict[keys[3]]=text_value
    #rating condition
    text_value:str=re.findall(r"给分好坏：(.*?)</li>",text_summury)[0]
    web_info_dict[keys[4]]=text_value
    #knowledge gain
    text_value:str=re.findall(r"收获大小：(.*?)</li>",text_summury)[0]
    web_info_dict[keys[5]]=text_value

    text_category:str=re.findall(\
      r"<table class=\"table table-condensed no-border\">(.*?)</table>",web)[0]
    #category for selection
    text_value:str=re.findall(\
      r"<td><strong>选课类别：</strong>(.*?)</td>",text_category)[0]
    web_info_dict[keys[6]]=text_value
    #category for teaching
    text_value:str=re.findall(\
      r"<td><strong>教学类型：</strong>(.*?)</td>",text_category)[0]
    web_info_dict[keys[7]]=text_value
    #category for curriculum
    text_value:str=re.findall(\
      r"<td><strong>课程类别：</strong>(.*?)</td>",text_category)[0]
    web_info_dict[keys[8]]=text_value
    #category for institude
    text_value:str=re.findall(\
      r"<td><strong>开课单位：</strong>(.*?)</td>",text_category)[0]
    web_info_dict[keys[9]]=text_value
    #category for level
    text_value:str=re.findall(\
      r"<td><strong>课程层次：</strong>(.*?)</td>",text_category)[0]
    web_info_dict[keys[10]]=text_value
    #credit amount
    text_value:str=re.findall(\
      r"</span>学分：</strong>(.*?)</td>",text_category)[0]
    web_info_dict[keys[11]]=text_value
  except:
    #if certain value is missing, but generally it works well
    return None

  if None in web_info_dict.values():  
    #if some value is missing 
    return None
  else:
    return web_info_dict

def get_course_list(web:str|None)->list[str]|None:
  '''get the urls of course detail pages from the main listing page\n
  return a list of individual urls'''
  if web==None:
    return None
  else:
    web=web.replace("\n","")

  url_list:list[str]=list()
  course_url_number_list:list[str]=\
    re.findall(r"<a class=\"px16\" href=\"/course/(\d*?)/\">",web)
  for number in course_url_number_list:
    new_url="https://icourse.club/course/"+number+"/"
    url_list.append(new_url)

  if len(url_list)==0:
    return None
  else:
    return url_list



