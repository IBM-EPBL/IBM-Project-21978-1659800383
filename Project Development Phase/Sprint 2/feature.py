# import regex    
# from tldextract import extract
# import ssl
# import socket
# from bs4 import BeautifulSoup
# import urllib.request
# import whois
# import datetime
# import requests
# import favicon
# import re
# import google
# from googlesearch import search

# #checking if URL contains any IP address. Returns -1 if contains else returns 1
# def having_IPhaving_IP_Address(url):
#      match=regex.search(
#    '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  #IPv4
#                     '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'  #IPv4 in hexadecimal
#                     '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}',url)     #Ipv6
#      if match:
#         #print match.group()
#         return -1
#      else:
#         #print 'No matching pattern found'
#         return 1

# #Checking for the URL length. Returns 1 (Legitimate) if the URL length is less than 54 characters
# #Returns 0 if the length is between 54 and 75
# #Else returns -1;
# def URLURL_Length (url):
#     length=len(url)
#     if(length<=75):
#          if(length<54):
#              return 1
#          else:
#              return 0
#     else:
#         return -1

# #Checking with the shortening URLs.
# #Returns -1 if any shortening URLs used.
# #Else returns 1
# def Shortining_Service (url):
#     match=regex.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
#                     'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
#                     'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
#                     'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
#                     'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
#                     'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
#                     'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net',url)
#     if match:
#         return -1
#     else:
#         return 1

# #Checking for @ symbol. Returns 1 if no @ symbol found. Else returns 0.
# def having_At_Symbol(url):
#     symbol=regex.findall(r'@',url)
#     if(len(symbol)==0):
#         return 1
#     else:
#         return -1 
    
# #Checking for Double Slash redirections. Returns -1 if // found. Else returns 1
# def double_slash_redirecting(url):
#     for i in range(8,len(url)):
#         if(url[i]=='/'):
            
#             if(url[i-1]=='/'):
#                 return -1
#     return 1

# #Checking for - in Domain. Returns -1 if '-' is found else returns 1.
# def Prefix_Suffix(url):
#     subDomain, domain, suffix = extract(url)
#     if(domain.count('-')):
#         return -1
#     else:
#         return  1

# #checking the Subdomain. Returns 1 if the subDomain contains less than 1 '.'
# #Returns 0 if the subDomain contains less than 2 '.'
# #Returns -1 if the subDomain contains more than 2 '.'
# def having_Sub_Domain(url):
#     subDomain, domain, suffix = extract(url)
#     if(subDomain.count('.')<=2):
#         if(subDomain.count('.')<=1):
#             return 1
#         else:
#             return 0
#     else:
#         return -1

# #Checking the SSL. Returns 1 if it returns the respomse code and -1 if exceptions are thrown.
# def SSLfinal_State(url):
#     try:
#         response = requests.get(url)
#         return 1
#     except Exception as e:
#         return -1
    
# #domains expires on ≤ 1 year returns -1, otherwise returns 1

# def Domain_registeration_length(url):
#     try:
#         domain = whois.whois(url)
#         exp=domain.expiration_date[0]
#         up=domain.updated_date[0]
#         domainlen=(exp-up).days
#         if(domainlen<=365):
#             return -1
#         else:
#             return 1
#     except:
#         return -1

# #Checking the Favicon. Returns 1 if the domain of the favicon image and the URL domain match else returns -1.
# def Favicon(url):
#     subDomain, domain, suffix = extract(url)
#     b=domain
#     try:
#         icons = favicon.get(url)
#         icon = icons[0]
#         subDomain, domain, suffix =extract(icon.url)
#         a=domain
#         if(a==b):
#             return 1
#         else:
#             return -1
#     except:
#         return -1    

# #Checking the Port of the URL. Returns 1 if the port is available else returns -1.
# def port(url):
#     try:
#         a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         location=(url[7:],80)
#         result_of_check = a_socket.connect_ex(location)
#         if result_of_check == 0:
#             return 1
#         else:
#             return -1
#         a_socket.close
#     except:
#         return -1

# # HTTPS token in part of domain of URL returns -1, otherwise returns 1
# def HTTPS_token(url):
#     match=re.search('https://|http://',url)
#     if (match.start(0)==0):
#         url=url[match.end(0):]
#     match=re.search('http|https',url)
#     if match:
#         return -1
#     else:
#         return 1
    
    
# #% of request URL<22% returns 1, otherwise returns -1
# def Request_URL(url):
#     try:
#         subDomain, domain, suffix = extract(url)
#         websiteDomain = domain
        
#         opener = urllib.request.urlopen(url).read()
#         soup = BeautifulSoup(opener, 'lxml')
#         imgs = soup.findAll('img', src=True)
#         total = len(imgs)
        
#         linked_to_same = 0
#         avg =0
#         for image in imgs:
#             subDomain, domain, suffix = extract(image['src'])
#             imageDomain = domain
#             if(websiteDomain==imageDomain or imageDomain==''):
#                 linked_to_same = linked_to_same + 1
#         vids = soup.findAll('video', src=True)
#         total = total + len(vids)
        
#         for video in vids:
#             subDomain, domain, suffix = extract(video['src'])
#             vidDomain = domain
#             if(websiteDomain==vidDomain or vidDomain==''):
#                 linked_to_same = linked_to_same + 1
#         linked_outside = total-linked_to_same
#         if(total!=0):
#             avg = linked_outside/total
            
#         if(avg<0.22):
#             return 1
#         else:
#             return -1
#     except:
#         return -1

# #:% of URL of anchor<31% returns 1, % of URL of anchor ≥ 31% and ≤ 67% returns 0, otherwise returns -1
# def URL_of_Anchor(url):
#     try:
#         subDomain, domain, suffix = extract(url)
#         websiteDomain = domain
        
#         opener = urllib.request.urlopen(url).read()
#         soup = BeautifulSoup(opener, 'lxml')
#         anchors = soup.findAll('a', href=True)
#         total = len(anchors)
#         linked_to_same = 0
#         avg = 0
#         for anchor in anchors:
#             subDomain, domain, suffix = extract(anchor['href'])
#             anchorDomain = domain
#             if(websiteDomain==anchorDomain or anchorDomain==''):
#                 linked_to_same = linked_to_same + 1
#         linked_outside = total-linked_to_same
#         if(total!=0):
#             avg = linked_outside/total
            
#         if(avg<0.31):
#             return 1
#         elif(0.31<=avg<=0.67):
#             return 0
#         else:
#             return -1
#     except:
#         return 0 
    
# #:% of links in <meta>, <script>and<link>tags < 25% returns 1, % of links in <meta>,
# #<script> and <link> tags ≥ 25% and ≤ 81% returns 0, otherwise returns -1
    
# def Links_in_tags(url):
#     try:
#         opener = urllib.request.urlopen(url).read()
#         soup = BeautifulSoup(opener, 'lxml')
        
#         no_of_meta =0
#         no_of_link =0
#         no_of_script =0
#         anchors=0
#         avg =0
#         for meta in soup.find_all('meta'):
#             no_of_meta = no_of_meta+1
#         for link in soup.find_all('link'):
#             no_of_link = no_of_link +1
#         for script in soup.find_all('script'):
#             no_of_script = no_of_script+1
#         for anchor in soup.find_all('a'):
#             anchors = anchors+1
#         total = no_of_meta + no_of_link + no_of_script+anchors
#         tags = no_of_meta + no_of_link + no_of_script
#         if(total!=0):
#             avg = tags/total

#         if(avg<0.25):
#             return -1
#         elif(0.25<=avg<=0.81):
#             return 0
#         else:
#             return 1        
#     except:        
#         return 0
 
# #Server Form Handling
# #SFH is "about: blank" or empty → phishing, SFH refers to a different domain → suspicious, otherwise → legitimate
# def SFH(url):
#     #ongoing
#     return -1

# #:using "mail()" or "mailto:" returning -1, otherwise returns 1
# def Submitting_to_email(url):
#     try:
#         opener = urllib.request.urlopen(url).read()
#         soup = BeautifulSoup(opener, 'lxml')
#         if(soup.find('mailto:','mail():')):
#             return -1
#         else:
#             return 1 
#     except:
#         return -1

# #Host name is not in URL returns -1, otherwise returns 1
# def Abnormal_URL(url):
#     subDomain, domain, suffix = extract(url)
#     try:
#         domain = whois.whois(url)
#         hostname=domain.domain_name[0].lower()
#         match=re.search(hostname,url)
#         if match:
#             return 1
#         else:
#             return -1
#     except:
#         return -1

# #number of redirect page ≤ 1 returns 1, otherwise returns 0
# def Redirect(url):
#     try:
#         request = requests.get(url)
#         a=request.history
#         if(len(a)<=1):
#             return 1
#         else:
#             return 0
        
#     except:
#         return 0
        

# #onMouseOver changes status bar returns -1, otherwise returns 1
# def on_mouseover(url):
#     try:
#         opener = urllib.request.urlopen(url).read()
#         soup = BeautifulSoup(opener, 'lxml')
        
#         no_of_script =0
#         for meta in soup.find_all(onmouseover=True):
#             no_of_script = no_of_script+1
#         if(no_of_script==0):
#             return 1
#         else:
#             return -1
#     except:
#         return -1
    
# #right click disabled returns -1, otherwise returns 1
# def RightClick(url):
#     try:
#         opener = urllib.request.urlopen(url).read()
#         soup = BeautifulSoup(opener, 'lxml')
#         if(soup.find_all('script',mousedown=True)):
#             return -1
#         else:
#             return 1
#     except:
#         return -1

# #popup window contains text field → phishing, otherwise → legitimate
# def popUpWidnow(url):
#     #ongoing
#     return 1

# #using iframe returns -1, otherwise returns 1
# def Iframe(url):
#     try:
#         opener = urllib.request.urlopen(url).read()
#         soup = BeautifulSoup(opener, 'lxml')
#         nmeta=0
#         for meta in soup.findAll('iframe',src=True):
#             nmeta= nmeta+1
#         if(nmeta!=0):
#             return -1
#         else:
#             return 1
#     except:
#         return -1


# #:age of domain ≥ 6 months returns 1, otherwise returns -1
# def age_of_domain(url):
#     try:
#         w = whois.whois(url).creation_date[0].year
#         if(w<=2018):
#             return 1
#         else:
#             return -1
#     except Exception as e:
#         return -1

# #no DNS record for domain returns -1, otherwise returns 1
# def DNSRecord(url):
    
#     subDomain, domain, suffix = extract(url)
#     try:
#         dns = 0
#         domain_name = whois.whois(url)
#     except:
#         dns = 1
        
#     if(dns == 1):
#         return -1
#     else:
#         return 1

# #website rank < 100.000 returns 1, website rank > 100.000 returns 0, otherwise returns -1
# def web_traffic(url):
#     try:
#         rank = BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + url).read(), "xml").find("REACH")['RANK']
#     except TypeError:
#         return -1
#     rank= int(rank)
#     if (rank<100000):
#         return 1
#     else:
#         return 0

# #:PageRank < 0,2 → phishing, otherwise → legitimate
# def Page_Rank(url):
#     #ongoing
#     return 1

# #webpage indexed by Google returns 1, otherwise returns -1
# def Google_Index(url):
#     try:
#         subDomain, domain, suffix = extract(url)
#         a=domain + '.' + suffix
#         query = url
#         for j in search(query, tld="co.in", num=5, stop=5, pause=2):
#             subDomain, domain, suffix = extract(j)
#             b=domain + '.' + suffix
#         if(a==b):
#             return 1
#         else:
#             return -1
#     except:
#         return -1


# #:number of links pointing to webpage = 0 returns 1, number of links pointing to webpage> 0
# #and ≤ 2 returns 0, otherwise returns -1

# def Links_pointing_to_page (url):
#     try:
#         opener = urllib.request.urlopen(url).read()
#         soup = BeautifulSoup(opener, 'lxml')
#         count = 0
#         for link in soup.find_all('a'):
#             count += 1
#         if(count>=2):
#             return 1
#         else:
#             return 0
#     except:
#         return -1

# #:host in top 10 phishing IPs or domains returns -1, otherwise returns 1
# def Statistical_report (url):
#     hostname = url
#     h = [(x.start(0), x.end(0)) for x in regex.finditer('https://|http://|www.|https://www.|http://www.', hostname)]
#     z = int(len(h))
#     if z != 0:
#         y = h[0][1]
#         hostname = hostname[y:]
#         h = [(x.start(0), x.end(0)) for x in regex.finditer('/', hostname)]
#         z = int(len(h))
#         if z != 0:
#             hostname = hostname[:h[0][0]]
#     url_match=regex.search('at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly',url)
#     try:
#         ip_address = socket.gethostbyname(hostname)
#         ip_match=regex.search('146\.112\.61\.108|213\.174\.157\.151|121\.50\.168\.88|192\.185\.217\.116|78\.46\.211\.158|181\.174\.165\.13|46\.242\.145\.103|121\.50\.168\.40|83\.125\.22\.219|46\.242\.145\.98|107\.151\.148\.44|107\.151\.148\.107|64\.70\.19\.203|199\.184\.144\.27|107\.151\.148\.108|107\.151\.148\.109|119\.28\.52\.61|54\.83\.43\.69|52\.69\.166\.231|216\.58\.192\.225|118\.184\.25\.86|67\.208\.74\.71|23\.253\.126\.58|104\.239\.157\.210|175\.126\.123\.219|141\.8\.224\.221|10\.10\.10\.10|43\.229\.108\.32|103\.232\.215\.140|69\.172\.201\.153|216\.218\.185\.162|54\.225\.104\.146|103\.243\.24\.98|199\.59\.243\.120|31\.170\.160\.61|213\.19\.128\.77|62\.113\.226\.131|208\.100\.26\.234|195\.16\.127\.102|195\.16\.127\.157|34\.196\.13\.28|103\.224\.212\.222|172\.217\.4\.225|54\.72\.9\.51|192\.64\.147\.141|198\.200\.56\.183|23\.253\.164\.103|52\.48\.191\.26|52\.214\.197\.72|87\.98\.255\.18|209\.99\.17\.27|216\.38\.62\.18|104\.130\.124\.96|47\.89\.58\.141|78\.46\.211\.158|54\.86\.225\.156|54\.82\.156\.19|37\.157\.192\.102|204\.11\.56\.48|110\.34\.231\.42',ip_address)  
#     except:
#         return -1

#     if url_match:
#         return -1
#     else:
#         return 1
    
# #returning scrapped data to calling function in app.py
# def main(url):


    
#     check = [[having_IPhaving_IP_Address (url),URLURL_Length(url),Shortining_Service(url),having_At_Symbol(url),
#              double_slash_redirecting(url),Prefix_Suffix(url),having_Sub_Domain(url),SSLfinal_State(url),
#               Domain_registeration_length(url),Favicon(url),port(url),HTTPS_token(url),Request_URL(url),
#               URL_of_Anchor(url),Links_in_tags(url),SFH(url),Submitting_to_email(url),Abnormal_URL(url),
#               Redirect(url),on_mouseover(url),RightClick(url),popUpWidnow(url),Iframe(url),
#               age_of_domain(url),DNSRecord(url),web_traffic(url),Page_Rank(url),Google_Index(url),
#               Links_pointing_to_page(url),Statistical_report(url)]]
    
    
#     print(check)
#     return check

import regex    
from tldextract import extract
import ssl
import socket
from bs4 import BeautifulSoup
import urllib.request
import whois
import datetime


def url_having_ip(url):
    symbol = regex.findall(r'(http((s)?)://)((((\d)+).)*)((\w)+)(/((\w)+))?@',url)
    if(len(symbol)!=0):
        having_ip = 1 #phishing
    else:
        having_ip = -1 #legitimate
    return(having_ip)
    return 0


def url_length(url):
    length=len(url)
    if(length<54):
        return -1
    elif(54<=length<=75):
        return 0
    else:
        return 1


def url_short(url):
    #ongoing
    return 0

def having_at_symbol(url):
    symbol=regex.findall(r'@',url)
    if(len(symbol)==0):
        return -1
    else:
        return 1 
    
def doubleSlash(url):
    #ongoing
    return 0

def prefix_suffix(url):
    subDomain, domain, suffix = extract(url)
    if(domain.count('-')):
        return 1
    else:
        return -1

def sub_domain(url):
    subDomain, domain, suffix = extract(url)
    if(subDomain.count('.')==0):
        return -1
    elif(subDomain.count('.')==1):
        return 0
    else:
        return 1

def SSLfinal_State(url):
    try:
#check wheather contains https       
        if(regex.search('^https',url)):
            usehttps = 1
        else:
            usehttps = 0
#getting the certificate issuer to later compare with trusted issuer 
        #getting host name
        subDomain, domain, suffix = extract(url)
        host_name = domain + "." + suffix
        context = ssl.create_default_context()
        sct = context.wrap_socket(socket.socket(), server_hostname = host_name)
        sct.connect((host_name, 443))
        certificate = sct.getpeercert()
        issuer = dict(x[0] for x in certificate['issuer'])
        certificate_Auth = str(issuer['commonName'])
        certificate_Auth = certificate_Auth.split()
        if(certificate_Auth[0] == "Network" or certificate_Auth == "Deutsche"):
            certificate_Auth = certificate_Auth[0] + " " + certificate_Auth[1]
        else:
            certificate_Auth = certificate_Auth[0] 
        trusted_Auth = ['Comodo','Symantec','GoDaddy','GlobalSign','DigiCert','StartCom','Entrust','Verizon','Trustwave','Unizeto','Buypass','QuoVadis','Deutsche Telekom','Network Solutions','SwissSign','IdenTrust','Secom','TWCA','GeoTrust','Thawte','Doster','VeriSign']        
#getting age of certificate
        startingDate = str(certificate['notBefore'])
        endingDate = str(certificate['notAfter'])
        startingYear = int(startingDate.split()[3])
        endingYear = int(endingDate.split()[3])
        Age_of_certificate = endingYear-startingYear
        
#checking final conditions
        if((usehttps==1) and (certificate_Auth in trusted_Auth) and (Age_of_certificate>=1) ):
            return -1 #legitimate
        elif((usehttps==1) and (certificate_Auth not in trusted_Auth)):
            return 0 #suspicious
        else:
            return 1 #phishing
        
    except Exception as e:
        
        return 1

def domain_registration(url):
    try:
        w = whois.whois(url)
        updated = w.updated_date
        exp = w.expiration_date
        length = (exp[0]-updated[0]).days
        if(length<=365):
            return 1
        else:
            return -1
    except:
        return 0

def favicon(url):
    #ongoing
    return 0

def port(url):
    #ongoing
    return 0

def https_token(url):
    subDomain, domain, suffix = extract(url)
    host =subDomain +'.' + domain + '.' + suffix 
    if(host.count('https')): #attacker can trick by putting https in domain part
        return 1
    else:
        return -1

def request_url(url):
    try:
        subDomain, domain, suffix = extract(url)
        websiteDomain = domain
        
        opener = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(opener, 'lxml')
        imgs = soup.findAll('img', src=True)
        total = len(imgs)
        
        linked_to_same = 0
        avg =0
        for image in imgs:
            subDomain, domain, suffix = extract(image['src'])
            imageDomain = domain
            if(websiteDomain==imageDomain or imageDomain==''):
                linked_to_same = linked_to_same + 1
        vids = soup.findAll('video', src=True)
        total = total + len(vids)
        
        for video in vids:
            subDomain, domain, suffix = extract(video['src'])
            vidDomain = domain
            if(websiteDomain==vidDomain or vidDomain==''):
                linked_to_same = linked_to_same + 1
        linked_outside = total-linked_to_same
        if(total!=0):
            avg = linked_outside/total
            
        if(avg<0.22):
            return -1
        elif(0.22<=avg<=0.61):
            return 0
        else:
            return 1
    except:
        return 0


def url_of_anchor(url):
    try:
        subDomain, domain, suffix = extract(url)
        websiteDomain = domain
        
        opener = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(opener, 'lxml')
        anchors = soup.findAll('a', href=True)
        total = len(anchors)
        linked_to_same = 0
        avg = 0
        for anchor in anchors:
            subDomain, domain, suffix = extract(anchor['href'])
            anchorDomain = domain
            if(websiteDomain==anchorDomain or anchorDomain==''):
                linked_to_same = linked_to_same + 1
        linked_outside = total-linked_to_same
        if(total!=0):
            avg = linked_outside/total
            
        if(avg<0.31):
            return -1
        elif(0.31<=avg<=0.67):
            return 0
        else:
            return 1
    except:
        return 0
    
def Links_in_tags(url):
    try:
        opener = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(opener, 'lxml')
        
        no_of_meta =0
        no_of_link =0
        no_of_script =0
        anchors=0
        avg =0
        for meta in soup.find_all('meta'):
            no_of_meta = no_of_meta+1
        for link in soup.find_all('link'):
            no_of_link = no_of_link +1
        for script in soup.find_all('script'):
            no_of_script = no_of_script+1
        for anchor in soup.find_all('a'):
            anchors = anchors+1
        total = no_of_meta + no_of_link + no_of_script+anchors
        tags = no_of_meta + no_of_link + no_of_script
        if(total!=0):
            avg = tags/total

        if(avg<0.25):
            return -1
        elif(0.25<=avg<=0.81):
            return 0
        else:
            return 1        
    except:        
        return 0

def sfh(url):
    #ongoing
    return 0

def email_submit(url):
    try:
        opener = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(opener, 'lxml')
        if(soup.find('mailto:')):
            return 1
        else:
            return -1 
    except:
        return 0

def abnormal_url(url):
    #ongoing
    return 0

def redirect(url):
    #ongoing
    return 0

def on_mouseover(url):
    #ongoing
    return 0

def rightClick(url):
    #ongoing
    return 0

def popup(url):
    #ongoing
    return 0

def iframe(url):
    #ongoing
    return 0

def age_of_domain(url):
    try:
        w = whois.whois(url)
        start_date = w.creation_date
        current_date = datetime.datetime.now()
        age =(current_date-start_date[0]).days
        if(age>=180):
            return -1
        else:
            return 1
    except Exception as e:
        print(e)
        return 0
        
def dns(url):
    #ongoing
    return 0

def web_traffic(url):
    #ongoing
    return 0

def page_rank(url):
    #ongoing
    return 0

def google_index(url):
    #ongoing
    return 0


def links_pointing(url):
    #ongoing
    return 0

def statistical(url):
    #ongoing
    return 0

def main(url):


    
    
    check = [[url_having_ip(url),url_length(url),url_short(url),having_at_symbol(url),
             doubleSlash(url),prefix_suffix(url),sub_domain(url),SSLfinal_State(url),
              domain_registration(url),favicon(url),port(url),https_token(url),request_url(url),
              url_of_anchor(url),Links_in_tags(url),sfh(url),email_submit(url),abnormal_url(url),
              redirect(url),on_mouseover(url),rightClick(url),popup(url),iframe(url),
              age_of_domain(url),dns(url),web_traffic(url),page_rank(url),google_index(url),
              links_pointing(url),statistical(url)]]
    
    
    
    return check

