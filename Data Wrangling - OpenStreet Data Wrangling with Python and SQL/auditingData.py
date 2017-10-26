# -*- coding:utf-8 -*-
'''
Auditing and Cleaning the OSM FILE
'''

import re
import xml.etree.cElementTree as ET
from collections import defaultdict
import pprint
import time

# OSM file
osm_file = "arcadia.osm"
update_file = "update_arcadia.osm"

# ================================================== #
#               Helper Functions                     #
# ================================================== #

def get_element(osm_file, tags=('node', 'way', 'relation')):
    "Takes as input osm file and tuple of nodes and yield nodes of types from tuple"
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:

            yield elem
            root.clear()


def timecall(fn, *args):
    '''
    Call function with args
    :param fn:
    :param args:
    :return:   the time in seconds and result.
    '''
    t0 = time.clock()
    result = fn(*args)
    t1 = time.clock()
    return t1 - t0, result

# ================================================== #
#          Auditing and Cleaning Data                #
# ================================================== #
street_type_mapping = { "St": "Street",
                        "St.": "Street",
                        "ST": "Street",
                        "Ave": "Avenue",
                        "Ave.": "Avenue",
                        "Av.": "Avenue",
                        "Av": "Avenue",
                        "Sq": "Square",
                        "CT": "Court",
                        "Ct": "Court",
                        "DR": "Drive",
                        "Dr": "Drive",
                        "Dr.": "Drive",
                        "Rd.": "Road",
                        "Rd": "Road",
                        "Pl": "Place",
                        "Hwy": "Highway",
                        "Ln": "Lane",
                        "Blvd": "Boulevard",
                        "Blvd.": "Boulevard",
                        "boulevard.": "Boulevard",
                        "Brdg" : "Bridge",
                        "Ter": "Terrace",
                        "Pkwy" : "Parkway"
                        }


def is_stree_name(element):
    return element.attrib['k'] == "addr:street"

def is_post_code(element):
    return element.attrib['k'] == "addr:postcode"

def is_phone(element):
    return element.attrib['k'] == "phone"


def audit_data(osm_file):
    '''
    Auditing the data in the three aspect of street_type,post_code,phone_type
    :param osm_file:
    :return: list which is issues
    '''

    result = {}
    audit_items =['street_type', 'post_code', 'phone']
    attr_street_types = defaultdict(set)
    bad_postcodes = []
    bad_phones = []

    with open(osm_file, 'r') as f:
        for event,element in ET.iterparse(f):
            if element.tag == "tag":
                val = element.attrib['v']
                # auditing street type
                if is_stree_name(element):
                   get_abbr_stree_type(attr_street_types,val)

                # auditing post code
                if is_post_code(element) and (not is_postcode_format(val)):
                    bad_postcodes.append(val)

                # auditing phone number
                if is_phone(element) and (not is_phone_format(val)):
                     bad_phones.append(val)

        result[audit_items[0]] = attr_street_types
        result[audit_items[1]] = bad_postcodes
        result[audit_items[2]] = bad_phones
    return result


def get_abbr_stree_type(abbr_street_types,street_name):
    '''
    verifying the street name whether has the abbreviation of street type
    :param abbr_street_types: defaultDict(set)
    :param street_name:
    '''

    expected_street_type = set(street_type_mapping.values())
    street_type_re = re.compile(r'\S+\.?$')

    m = street_type_re.search(street_name)

    if m:
        street_type = m.group()
        if street_type not in expected_street_type:
            abbr_street_types[street_type].add(street_name)


def is_postcode_format(postcode):
    '''
    Verifying the format of the post code  is correct such as 91007
    :param postcode:
    :return:
    '''
    return re.compile(r'.*(\d{5}(\-\d{4})?)$').match(postcode)

def is_phone_format(phone):
    '''
    Verfying the format of the phone is correct just like (626)000-0000
    :param phone:
    :return:
    '''
    return re.compile(r'[(][\d]{3}[)][\d]{3}-[\d]{4}').match(phone)


"printing results"
# pprint.pprint(timecall(audit_data, osm_file))




def update_data_to_new_file(old_file, new_file):

    last_word = lambda x: x.split()[-1]

    with open(new_file, 'wb') as output:
        output.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        output.write(b'<osm>\n  ')
        for i, element in enumerate(get_element(old_file)):

            for child_elem in element.iter('tag'):
                # modify street name
                if is_stree_name(child_elem):

                    street_name = child_elem.attrib['v']
                    if last_word(street_name) in street_type_mapping.keys():
                        street_name = street_name.replace(last_word(street_name), street_type_mapping[last_word(street_name)])
                        child_elem.set('v',street_name)

                # modify phone
                if is_phone(child_elem):
                    phone =child_elem.attrib['v']
                    child_elem.set('v', phone_format(phone))


            output.write(ET.tostring(element, encoding='utf-8'))

        output.write(b'</osm>')

def phone_format(phone):
    if not is_phone_format(phone):
        # remove +()-.' '
        phone = re.sub('[+()-.]', '', phone).replace(' ', '')

        # process letter in phone number
        intab  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        outtab = "22233344455566677778889999"
        transtab = str.maketrans(intab, outtab)
        phone = phone.translate(transtab)

        # format
        n = 0
        if len(phone) == 11:
            n = 1
        phone = '(%s)%s-%s' % (phone[n:n+3], phone[n+3:n+6], phone[n+6:])


    return phone

print(phone_format('(626)377-YOGA'))

# update_data_to_new_file(osm_file,update_file)
"checking the data of new file whether has incorrect data"
# pprint.pprint(audit_data(update_file))
