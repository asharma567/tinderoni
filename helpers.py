import re
import robobrowser
from IPython.display import Image
from IPython.display import display
import joblib
import time

MOBILE_USER_AGENT = "Mozilla/5.0 (Linux; U; en-gb; KFTHWI Build/JDQ39) AppleWebKit/535.19 (KHTML, like Gecko) Silk/3.16 Safari/535.19"
FB_AUTH = "https://www.facebook.com/v2.6/dialog/oauth?redirect_uri=fb464891386855067%3A%2F%2Fauthorize%2F&display=touch&state=%7B%22challenge%22%3A%22IUUkEUqIGud332lfu%252BMJhxL4Wlc%253D%22%2C%220_auth_logger_id%22%3A%2230F06532-A1B9-4B10-BB28-B29956C71AB1%22%2C%22com.facebook.sdk_client_state%22%3Atrue%2C%223_method%22%3A%22sfvc_auth%22%7D&scope=user_birthday%2Cuser_photos%2Cuser_education_history%2Cemail%2Cuser_relationship_details%2Cuser_friends%2Cuser_work_history%2Cuser_likes&response_type=token%2Csigned_request&default_audience=friends&return_scopes=true&auth_type=rerequest&client_id=464891386855067&ret=login&sdk=ios&logger_id=30F06532-A1B9-4B10-BB28-B29956C71AB1&ext=1470840777&hash=AeZqkIcf-NEW6vBd"

def get_access_token(email, password):

    s = robobrowser.RoboBrowser(user_agent=MOBILE_USER_AGENT, parser="lxml")
    s.open(FB_AUTH)

    ##submit login form##
    f = s.get_form()
    f["pass"] = password
    f["email"] = email
    s.submit_form(f)

    ##click the 'ok' button on the dialog informing you that you have already authenticated with the Tinder app##
    f = s.get_form()
    s.submit_form(f, submit=f.submit_fields['__CONFIRM__'])

    ##get access token from the html response##
    access_token = re.search(r"access_token=([\w\d]+)", s.response.content.decode()).groups()[0]

    return access_token

def like_my_likes(user_dict, my_top_likes, super=False, dislike=False):
    
    for index, user_name_tuple in user_dict.items():
        name, user_object_from_pynder = user_name_tuple
        
        
        if index in my_top_likes:
            if super:
                print( user_object_from_pynder.superlike())
            else:
                if not dislike:
                    print( user_object_from_pynder.like())
                else:
                    print( user_object_from_pynder.dislike())


def dump_images_of_my_likes(user_dict, my_top_likes, path_to='tinder_pics_likes/'):
    
    for index, user_name_tuple in user_dict.items():
        name, user_object_from_pynder = user_name_tuple
        if index in my_top_likes: 
            _write_to_jpeg(user_object_from_pynder.photos, name, path_to)
    return None

def dump_objects_of_my_likes(user_dict, my_top_likes, path_to):
    try:
        list_of_users_objs = joblib.load(path_to)
    except:
        list_of_users_objs = []    
    
    for index, user_name_tuple in user_dict.items():
        name_of_user, user_object_from_pynder = user_name_tuple
        
        join_w_images_key = str(int(time.time())) + '_' + name_of_user + '_' + str(index)
        prefs = [
            user_object_from_pynder.name,
            user_object_from_pynder.age,
            user_object_from_pynder.schools,
            user_object_from_pynder.bio,
            user_object_from_pynder.jobs,
            user_object_from_pynder.distance_km, 
            len(user_object_from_pynder.common_connections),
            user_object_from_pynder.instagram_username,
            join_w_images_key
        ]

        if index in my_top_likes: 
            list_of_users_objs.append(prefs)
            joblib.dump(list_of_users_objs, path_to)

    return None


def _write_to_jpeg(list_of_user_photos, name_of_user, path_to):
    for idx, photo in enumerate(list_of_user_photos):
        filename_str = str(int(time.time())) + '_' + name_of_user + '_' + str(idx) + '.jpeg'
        with open(path_to + filename_str, 'wb') as f:
            try:
                f.write(Image(photo, width=600, height=600).data)
            except:
                print( 'error writing the pic for: ' + name_of_user )
                continue

def show_imgs(list_of_images):
    for pic in list_of_images:
        display(Image(pic, width=300, height=300))
    return None

def dump_text_data_of_my_likes(user_dict, my_top_likes, path_to='tinder_likes.csv'):

    for index, user_name_tuple in user_dict.items():
        
        if index in my_top_likes: 
            
            name, user_object_from_pynder = user_name_tuple
            
            _write_to_csv(
                    user_object_from_pynder.distance_km, 
                    user_object_from_pynder.age,
                    user_object_from_pynder.bio,
                    user_object_from_pynder.schools,
                    user_object_from_pynder.jobs,
                    name, 
                    path_to
                )
    return None

def _write_to_csv(distance, age, bio, schools, jobs, name_of_user, path_to):

    write_time = str(int(time.time()))
    try:
        a_line = '|'.join([
            str(distance), 
            str(age), 
            special_char_translation(bio), 
            special_char_translation(','.join(schools)) if schools != [] else 'na', 
            special_char_translation(','.join(jobs)) if jobs != [] else 'na', 
            name_of_user, 
            write_time
            ])
    except:
        print( schools, jobs        )
    with open(path_to, 'ab') as f:
        try:
            f.write(a_line)
            f.write('\n')
        except:
            print( 'error writing the text for: ' + name_of_user )
            
    return None


if __name__ == '__main__':
    #EXAMPLE 
    import pynder
    
    email, password = '8184223740', 'getmelaid'
    legit_token = get_access_token(email, password)
    session = pynder.Session(facebook_token=legit_token)

    users={}
    for index, user in enumerate(session.nearby_users()):
        
        print( index, user.name)
        users[index] = (user.name, user)
        show_imgs(user.photos)
        
        if index > 10: break

    list_of_likes = [1, 2, 3]

    #fix the None error
    dump_images_of_my_likes(users, list_of_likes, 'test_directory/')
    dump_text_data_of_my_likes(users, list_of_likes, 'test_directory/')
    like_my_likes(users, list_of_likes)
    