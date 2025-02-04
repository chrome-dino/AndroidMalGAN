from lxml import etree
import os
import pandas as pd
from collections import Counter
import json
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler

INTENT_STORE = '../intent_store'
INTENT_STORE_LIMIT = 50
PERM_STORE = '../perm_store'
PERM_STORE_LIMIT = 50
MAX_COLLECT = 100
LIMIT = False
API_STORE = '../api_store'
API_STORE_LIMIT = 50


# def labeled_permissions_data(root_dir='.', permissions_file='permissions.txt'):
#     permissions = []
#     # https://github.com/mrcse/Android-Permissions/blob/main/xml_permissions.txt
#     with open(permissions_file, 'r') as fp:
#         read_lines = fp.readlines()
#         for line in read_lines:
#             permissions.append(line.rstrip())
#
#     parser = etree.XMLParser(recover=True)
#     for root, dirs, files in os.walk(root_dir):  # Scanning through each file in each subdirectory
#         for file in files:
#             if file == 'AndroidManifest.xml':
#                 file_dest = os.path.join(root, file)
#                 try:
#                     root = etree.parse(file_dest, parser=parser)
#                 except etree.XMLSyntaxError:
#                     continue
#                 try:
#                     file_permissions = root.findall("uses-permission")
#                 except AssertionError:
#                     continue
#
#                 row = dict.fromkeys(permissions, 0)
#                 for perm in file_permissions:
#                     for att in perm.attrib:
#                         p = perm.attrib[att]
#                         if p in row.keys():
#                             row[p] = 1
#
#                 return [row]


# def extract_permission_features(root_dir='../samples', malware=False):
#     if malware:
#         if os.path.isfile('malware_permissions.csv'):
#             os.remove('malware_permissions.csv')
#     else:
#         if os.path.isfile('benign_permissions.csv'):
#             os.remove('benign_permissions.csv')
#     permissions = []
#     # https://github.com/mrcse/Android-Permissions/blob/main/xml_permissions.txt
#     with open('permissions.txt', 'r') as fp:
#         read_lines = fp.readlines()
#         for line in read_lines:
#             permissions.append(line.rstrip())
#
#     count = 0
#     parser = etree.XMLParser(recover=True)
#     for root, dirs, files in os.walk(root_dir):  # Scanning through each file in each subdirectory
#         for file in files:
#             if file == 'AndroidManifest.xml':
#                 file_dest = os.path.join(root, file)
#                 try:
#                     root = etree.parse(file_dest, parser=parser)
#                 except etree.XMLSyntaxError:
#                     continue
#                 try:
#                     file_permissions = root.findall("uses-permission")
#                 except AssertionError:
#                     continue
#
#                 row = dict.fromkeys(permissions, 0)
#                 for perm in file_permissions:
#                     for att in perm.attrib:
#                         p = perm.attrib[att]
#                         if p in row.keys():
#                             row[p] = 1
#                 if malware:
#                     if os.path.isfile('malware_permissions.csv'):
#                         df = pd.DataFrame([row])
#                         df.to_csv('malware_permissions.csv', mode='a', header=False)
#                     else:
#                         df = pd.DataFrame([row])
#                         df.to_csv('malware_permissions.csv')
#                 else:
#                     if os.path.isfile('benign_permissions.csv'):
#                         df = pd.DataFrame([row])
#                         df.to_csv('benign_permissions.csv', mode='a', header=False)
#                     else:
#                         df = pd.DataFrame([row])
#                         df.to_csv('benign_permissions.csv')
#
#                 count += 1
#                 if count % 100 == 0:
#                     print(f'finished extracting {str(count)} files')
#
#                 if LIMIT:
#                     if count >= MAX_COLLECT:
#                         break


def extract_perms_features(root_dir='.', feature_count=300, exclude=None):
    parser = etree.XMLParser(recover=True)
    count = 0
    perm_store_count = 0
    perm_dict_group = {}
    filelist = [f for f in os.listdir(PERM_STORE) if f.endswith(".json")]
    for f in filelist:
        os.remove(os.path.join(PERM_STORE, f))
    for root, dirs, files in os.walk(root_dir):  # Scanning through each file in each subdirectory
        for file in files:
            if file == 'AndroidManifest.xml':
                perm_dict = {}
                file_dest = os.path.join(root, file)
                try:
                    xml_root = etree.parse(file_dest, parser=parser)
                except etree.XMLSyntaxError:
                    continue
                try:
                    file_permissions = xml_root.findall("uses-permission")
                except AssertionError:
                    continue
                for perm in file_permissions:
                    for att in perm.attrib:
                        if 'name' not in att:
                            continue
                        p = perm.attrib[att]
                        if exclude and p in exclude:
                            continue
                        perm_dict[p] = 1
                scaler = MinMaxScaler()
                scaled = np.array(list(perm_dict.items()), dtype=object)
                if not scaled.any():
                    count += 1
                    if count % 100 == 0:
                        print(f'finished extracting {str(count)} files')

                    if count % PERM_STORE_LIMIT == 0:
                        with open(f'./{PERM_STORE}/perm_store_{str(perm_store_count)}.json', 'w') as f:
                            json.dump(perm_dict_group, f)
                        perm_dict_group = {}
                        perm_store_count += 1
                    continue

                scaled[:, 1] = (scaled[:, -1].reshape(1, scaled.shape[0]).T).T
                # scaled[:, 1] = scaler.fit_transform(scaled[:, -1].reshape(1, scaled.shape[0]).T).T
                perm_dict = dict(scaled)
                perm_dict = {key: val for key, val in perm_dict.items() if val != 0}
                perm_dict_group = {x: perm_dict_group.get(x, 0) + perm_dict.get(x, 0) for x in
                              set(perm_dict_group).union(perm_dict)}
                count += 1
                # current_hash = re.findall(r"([a-fA-F\d]{32})", file_dest)[0]
                if count % 100 == 0:
                    print(f'finished extracting {str(count)} files')

                if count % PERM_STORE_LIMIT == 0:
                    with open(f'./{PERM_STORE}/perm_store_{str(perm_store_count)}.json', 'w') as f:
                        json.dump(perm_dict_group, f)
                    perm_dict_group = {}
                    perm_store_count += 1
                if LIMIT:
                    if count >= MAX_COLLECT:
                        break
        if LIMIT:
            if count >= MAX_COLLECT:
                break

    if perm_dict_group:
        with open(f'./{PERM_STORE}/perm_store_{str(perm_store_count)}.json', 'w') as f:
            json.dump(perm_dict_group, f)
    perm_dict = {}
    for f in os.listdir(PERM_STORE):
        if not f.endswith('.json'):
            continue
        with open(os.path.join(PERM_STORE, f)) as json_file:
            data = json.load(json_file)
            perm_dict = {x: perm_dict.get(x, 0) + data.get(x, 0) for x in set(perm_dict).union(data)}
    filtered_perm = Counter(perm_dict).most_common(feature_count)
    # print('finished extracting perms')
    return [filtered_perm[n][0] for n in range(len(filtered_perm))]


def labeled_perm_data(root_dir='.', perm_features=None, malware=False, single_file=False):
    if not single_file:
        if malware:
            if os.path.isfile('malware_permissions.csv'):
                os.remove('malware_permissions.csv')
        else:
            if os.path.isfile('benign_permissions.csv'):
                os.remove('benign_permissions.csv')

    parser = etree.XMLParser(recover=True)
    count = 0
    for root, dirs, files in os.walk(root_dir):  # Scanning through each file in each subdirectory
        for file in files:
            if file == 'AndroidManifest.xml':
                row = dict.fromkeys(perm_features, 0)
                file_dest = os.path.join(root, file)
                try:
                    root = etree.parse(file_dest, parser=parser)
                except etree.XMLSyntaxError:
                    continue
                try:
                    file_permissions = root.findall('uses-permission')
                except AssertionError:
                    continue

                for perm in file_permissions:
                    for att in perm.attrib:
                        p = perm.attrib[att]
                        if p in perm_features:
                            row[p] += 1

                if not single_file:
                    if malware:
                        if os.path.isfile('malware_permissions.csv'):
                            df = pd.DataFrame([row])
                            df.to_csv('malware_permissions.csv', mode='a', header=False)
                        else:
                            df = pd.DataFrame([row])
                            df.to_csv('malware_permissions.csv')
                    else:
                        if os.path.isfile('benign_permissions.csv'):
                            df = pd.DataFrame([row])
                            df.to_csv('benign_permissions.csv', mode='a', header=False)
                        else:
                            df = pd.DataFrame([row])
                            df.to_csv('benign_permissions.csv')
                else:
                    return [row]

                count += 1
                if count % 100 == 0:
                    print(f'finished labelling {str(count)} files')

                if LIMIT:
                    if count >= MAX_COLLECT:
                        return


def extract_intents_features(root_dir='.', feature_count=300, exclude=None):
    parser = etree.XMLParser(recover=True)
    count = 0
    intent_store_count = 0
    intent_dict_group = {}
    filelist = [f for f in os.listdir(INTENT_STORE) if f.endswith(".json")]
    for f in filelist:
        os.remove(os.path.join(INTENT_STORE, f))
    for root, dirs, files in os.walk(root_dir):  # Scanning through each file in each subdirectory
        for file in files:
            if file == 'AndroidManifest.xml':
                intent_dict = {}
                file_dest = os.path.join(root, file)
                try:
                    root = etree.parse(file_dest, parser=parser)
                except etree.XMLSyntaxError:
                    continue
                try:
                    applications = root.findall('application')
                except AssertionError:
                    continue
                for application in applications:
                    for activity in application.findall('activity'):
                        for intent in activity.findall('intent-filter'):
                            actions = intent.findall('action')
                            categories = intent.findall('category')
                            for action in actions:
                                for att in action.attrib:
                                    if 'name' not in att:
                                        continue
                                    a = action.attrib[att]
                                    if exclude and a in exclude:
                                        continue
                                    intent_dict[a] = 1
                                    # if a in intent_dict.keys():
                                    #     intent_dict[a] += 1
                                    # else:
                                    #     intent_dict[a] = 1
                            # for category in categories:
                            #     for att in category.attrib:
                            #         if 'name' not in att:
                            #             continue
                            #         c = category.attrib[att]
                            #         if exclude and c in exclude:
                            #             continue
                            #         intent_dict[c] = 1
                                    # if c in intent_dict.keys():
                                    #     intent_dict[c] += 1
                                    # else:
                                    #     intent_dict[c] = 1

                    for activity_alias in application.findall('activity-alias'):
                        for intent in activity_alias.findall('intent-filter'):
                            actions = intent.findall('action')
                            categories = intent.findall('category')
                            for action in actions:
                                for att in action.attrib:
                                    if 'name' not in att:
                                        continue
                                    a = action.attrib[att]
                                    if exclude and a in exclude:
                                        continue
                                    intent_dict[a] = 1
                                    # if a in intent_dict.keys():
                                    #     intent_dict[a] += 1
                                    # else:
                                    #     intent_dict[a] = 1
                            # for category in categories:
                            #     for att in category.attrib:
                            #         if 'name' not in att:
                            #             continue
                            #         c = category.attrib[att]
                            #         if exclude and c in exclude:
                            #             continue
                            #         intent_dict[c] = 1
                                    # if c in intent_dict.keys():
                                    #     intent_dict[c] += 1
                                    # else:
                                    #     intent_dict[c] = 1

                    for provider in application.findall('provider'):
                        for intent in provider.findall('intent-filter'):
                            actions = intent.findall('action')
                            categories = intent.findall('category')
                            for action in actions:
                                for att in action.attrib:
                                    if 'name' not in att:
                                        continue
                                    a = action.attrib[att]
                                    if exclude and a in exclude:
                                        continue
                                    intent_dict[a] = 1
                                    # if a in intent_dict.keys():
                                    #     intent_dict[a] += 1
                                    # else:
                                    #     intent_dict[a] = 1
                            # for category in categories:
                            #     for att in category.attrib:
                            #         if 'name' not in att:
                            #             continue
                            #         c = category.attrib[att]
                            #         if exclude and c in exclude:
                            #             continue
                            #         intent_dict[c] = 1
                                    # if c in intent_dict.keys():
                                    #     intent_dict[c] += 1
                                    # else:
                                    #     intent_dict[c] = 1

                    for receiver in application.findall('receiver'):
                        for intent in receiver.findall('intent-filter'):
                            actions = intent.findall('action')
                            categories = intent.findall('category')
                            for action in actions:
                                for att in action.attrib:
                                    if 'name' not in att:
                                        continue
                                    a = action.attrib[att]
                                    if exclude and a in exclude:
                                        continue
                                    intent_dict[a] = 1
                                    # if a in intent_dict.keys():
                                    #     intent_dict[a] += 1
                                    # else:
                                    #     intent_dict[a] = 1
                            # for category in categories:
                            #     for att in category.attrib:
                            #         if 'name' not in att:
                            #             continue
                            #         c = category.attrib[att]
                            #         if exclude and c in exclude:
                            #             continue
                            #         intent_dict[c] = 1
                                    # if c in intent_dict.keys():
                                    #     intent_dict[c] += 1
                                    # else:
                                    #     intent_dict[c] = 1

                    for service in application.findall('service'):
                        for intent in service.findall('intent-filter'):
                            actions = intent.findall('action')
                            categories = intent.findall('category')
                            for action in actions:
                                for att in action.attrib:
                                    if 'name' not in att:
                                        continue
                                    a = action.attrib[att]
                                    if exclude and a in exclude:
                                        continue
                                    intent_dict[a] = 1
                                    # if a in intent_dict.keys():
                                    #     intent_dict[a] += 1
                                    # else:
                                    #     intent_dict[a] = 1
                            # for category in categories:
                            #     for att in category.attrib:
                            #         if 'name' not in att:
                            #             continue
                            #         c = category.attrib[att]
                            #         if exclude and c in exclude:
                            #             continue
                            #         intent_dict[c] = 1
                                    # if c in intent_dict.keys():
                                    #     intent_dict[c] += 1
                                    # else:
                                    #     intent_dict[c] = 1

                scaler = MinMaxScaler()
                scaled = np.array(list(intent_dict.items()), dtype=object)
                if not scaled.any():
                    count += 1
                    # current_hash = re.findall(r"([a-fA-F\d]{32})", file_dest)[0]
                    if count % 100 == 0:
                        print(f'finished extracting {str(count)} files')

                    if count % INTENT_STORE_LIMIT == 0:
                        with open(f'./{INTENT_STORE}/intent_store_{str(intent_store_count)}.json', 'w') as f:
                            json.dump(intent_dict_group, f)
                        intent_dict_group = {}
                        intent_store_count += 1
                    continue
                scaled[:, 1] = (scaled[:, -1].reshape(1, scaled.shape[0]).T).T
                # scaled[:, 1] = scaler.fit_transform(scaled[:, -1].reshape(1, scaled.shape[0]).T).T
                intent_dict = dict(scaled)
                intent_dict = {key: val for key, val in intent_dict.items() if val != 0}
                intent_dict_group = {x: intent_dict_group.get(x, 0) + intent_dict.get(x, 0) for x in
                              set(intent_dict_group).union(intent_dict)}
                count += 1
                # current_hash = re.findall(r"([a-fA-F\d]{32})", file_dest)[0]
                if count % 100 == 0:
                    print(f'finished extracting {str(count)} files')

                if count % INTENT_STORE_LIMIT == 0:
                    with open(f'./{INTENT_STORE}/intent_store_{str(intent_store_count)}.json', 'w') as f:
                        json.dump(intent_dict_group, f)
                    intent_dict_group = {}
                    intent_store_count += 1
                if LIMIT:
                    if count >= MAX_COLLECT:
                        break
        if LIMIT:
            if count >= MAX_COLLECT:
                break

    if intent_dict_group:
        with open(f'./{INTENT_STORE}/intent_store_{str(intent_store_count)}.json', 'w') as f:
            json.dump(intent_dict_group, f)
    intent_dict = {}
    for f in os.listdir(INTENT_STORE):
        if not f.endswith('.json'):
            continue
        with open(os.path.join(INTENT_STORE, f)) as json_file:
            data = json.load(json_file)
            intent_dict = {x: intent_dict.get(x, 0) + data.get(x, 0) for x in set(intent_dict).union(data)}
    filtered_intents = Counter(intent_dict).most_common(feature_count)
    # print('finished extracting intents')
    return [filtered_intents[n][0] for n in range(len(filtered_intents))]


def labeled_intent_data(root_dir='.', intent_features=None, malware=False, single_file=False):
    if not single_file:
        if malware:
            if os.path.isfile('malware_intents.csv'):
                os.remove('malware_intents.csv')
        else:
            if os.path.isfile('benign_intents.csv'):
                os.remove('benign_intents.csv')

    parser = etree.XMLParser(recover=True)
    count = 0
    for root, dirs, files in os.walk(root_dir):  # Scanning through each file in each subdirectory
        for file in files:
            if file == 'AndroidManifest.xml':
                row = dict.fromkeys(intent_features, 0)
                file_dest = os.path.join(root, file)
                try:
                    root = etree.parse(file_dest, parser=parser)
                except etree.XMLSyntaxError:
                    continue
                try:
                    applications = root.findall('application')
                except AssertionError:
                    continue
                for application in applications:
                    for activity in application.findall('activity'):
                        for intent in activity.findall('intent-filter'):
                            actions = intent.findall('action')
                            categories = intent.findall('category')
                            for action in actions:
                                for att in action.attrib:
                                    a = action.attrib[att]
                                    if a in intent_features:
                                        row[a] += 1
                            for category in categories:
                                for att in category.attrib:
                                    c = category.attrib[att]
                                    if c in intent_features:
                                        row[c] += 1

                    for activity_alias in application.findall('activity-alias'):
                        for intent in activity_alias.findall('intent-filter'):
                            actions = intent.findall('action')
                            categories = intent.findall('category')
                            for action in actions:
                                for att in action.attrib:
                                    a = action.attrib[att]
                                    if a in intent_features:
                                        row[a] += 1
                            for category in categories:
                                for att in category.attrib:
                                    c = category.attrib[att]
                                    if c in intent_features:
                                        row[c] += 1

                    for provider in application.findall('provider'):
                        for intent in provider.findall('intent-filter'):
                            actions = intent.findall('action')
                            categories = intent.findall('category')
                            for action in actions:
                                for att in action.attrib:
                                    a = action.attrib[att]
                                    if a in intent_features:
                                        row[a] += 1
                            for category in categories:
                                for att in category.attrib:
                                    c = category.attrib[att]
                                    if c in intent_features:
                                        row[c] += 1

                    for receiver in application.findall('receiver'):
                        for intent in receiver.findall('intent-filter'):
                            actions = intent.findall('action')
                            categories = intent.findall('category')
                            for action in actions:
                                for att in action.attrib:
                                    a = action.attrib[att]
                                    if a in intent_features:
                                        row[a] += 1
                            for category in categories:
                                for att in category.attrib:
                                    c = category.attrib[att]
                                    if c in intent_features:
                                        row[c] += 1

                    for service in application.findall('service'):
                        for intent in service.findall('intent-filter'):
                            actions = intent.findall('action')
                            categories = intent.findall('category')
                            for action in actions:
                                for att in action.attrib:
                                    a = action.attrib[att]
                                    if a in intent_features:
                                        row[a] += 1
                            for category in categories:
                                for att in category.attrib:
                                    c = category.attrib[att]
                                    if c in intent_features:
                                        row[c] += 1
                if not single_file:
                    if malware:
                        if os.path.isfile('malware_intents.csv'):
                            df = pd.DataFrame([row])
                            df.to_csv('malware_intents.csv', mode='a', header=False)
                        else:
                            df = pd.DataFrame([row])
                            df.to_csv('malware_intents.csv')
                    else:
                        if os.path.isfile('benign_intents.csv'):
                            df = pd.DataFrame([row])
                            df.to_csv('benign_intents.csv', mode='a', header=False)
                        else:
                            df = pd.DataFrame([row])
                            df.to_csv('benign_intents.csv')
                else:
                    return [row]

                count += 1
                if count % 100 == 0:
                    print(f'finished labelling {str(count)} files')

                if LIMIT:
                    if count >= MAX_COLLECT:
                        return


def extract_api_calls(smali_file):
    api_calls = []
    with open(smali_file, 'r') as file:
        for line in file:
            match = re.search(r'invoke-\w+ {.*}, (L[\w/;]+;->[\w$]+)\(.*\)', line)
            if match:
                api_calls.append(match.group(1))
    return api_calls


def extract_api_features(root_dir='../samples', feature_count=300, exclude=None):
    count = 0
    api_store_count = 0
    current_hash = ''
    api_dict = {}
    api_dict_group = {}
    first = True
    filelist = [f for f in os.listdir(API_STORE) if f.endswith(".json")]
    for f in filelist:
        os.remove(os.path.join(API_STORE, f))
    for root, dirs, files in os.walk(root_dir):  # Scanning through each file in each subdirectory
        for file in files:
            if file.endswith(".smali"):
                file_dest = os.path.join(root, file)
                md5_hash = re.findall(r"([a-fA-F\d]{32})", file_dest)[0]
                if first:
                    current_hash = md5_hash
                    first = False
                if md5_hash != current_hash:

                    scaler = MinMaxScaler()
                    scaled = np.array(list(api_dict.items()), dtype=object)
                    if not scaled.any():
                        api_dict = {}
                        count += 1
                        if count % 100 == 0:
                            print(f'finished extracting {str(count)} files')

                        current_hash = md5_hash
                        print(current_hash + ' ' + str(count))

                        if count % API_STORE_LIMIT == 0:
                            with open(f'./{API_STORE}/api_store_{str(api_store_count)}.json', 'w') as f:
                                json.dump(api_dict_group, f)
                            api_dict_group = {}
                            api_store_count += 1
                        continue
                    scaled[:, 1] = scaler.fit_transform(scaled[:, -1].reshape(1, scaled.shape[0]).T).T
                    api_dict = dict(scaled)
                    api_dict = {key: val for key, val in api_dict.items() if val != 0}
                    api_dict_group = {x: api_dict_group.get(x, 0) + api_dict.get(x, 0) for x in
                                         set(api_dict_group).union(api_dict)}
                    api_dict = {}
                    count += 1
                    if count % 100 == 0:
                        print(f'finished extracting {str(count)} files')

                    current_hash = md5_hash
                    print(current_hash + ' ' + str(count))

                    if count % API_STORE_LIMIT == 0:
                        with open(f'./{API_STORE}/api_store_{str(api_store_count)}.json', 'w') as f:
                            json.dump(api_dict_group, f)
                        api_dict_group = {}
                        api_store_count += 1
                    if LIMIT:
                        if count >= MAX_COLLECT:
                            break

                ####################################################################################################
                api_calls = extract_api_calls(file_dest)
                for api_call in api_calls:
                    if exclude and api_call in exclude:
                        continue
                    if api_call in api_dict.keys():
                        api_dict[api_call] += 1
                    else:
                        api_dict[api_call] = 1
        if LIMIT:
            if count >= MAX_COLLECT:
                break

    if api_dict_group:
        with open(f'./{API_STORE}/api_store_{str(api_store_count)}.json', 'w') as f:
            json.dump(api_dict_group, f)
    api_dict = {}
    for f in os.listdir(API_STORE):
        if not f.endswith('.json'):
            continue
        with open(os.path.join(API_STORE, f)) as json_file:
            data = json.load(json_file)
            api_dict = {x: api_dict.get(x, 0) + data.get(x, 0) for x in set(api_dict).union(data)}

    filtered_apis = Counter(api_dict).most_common(feature_count)
    print('finished extracting apis')
    return [filtered_apis[n][0] for n in range(len(filtered_apis))]


def labeled_api_data(root_dir='.', api_features=None, malware=False, single_file=False):
    if not single_file:
        if malware:
            if os.path.isfile('malware_apis.csv'):
                os.remove('malware_apis.csv')
        else:
            if os.path.isfile('benign_apis.csv'):
                os.remove('benign_apis.csv')

    count = 0
    current_hash = ''
    first = True
    row = dict.fromkeys(api_features, 0)
    for root, dirs, files in os.walk(root_dir):  # Scanning through each file in each subdirectory
        for file in files:
            if file.endswith(".smali"):
                file_dest = os.path.join(root, file)
                md5_hash = re.findall(r"([a-fA-F\d]{32})", file_dest)[0]
                if first:
                    current_hash = md5_hash
                    first = False
                if md5_hash != current_hash:
                    if not single_file:
                        if malware:
                            if os.path.isfile('malware_apis.csv'):
                                df = pd.DataFrame([row])
                                df.to_csv('malware_apis.csv', mode='a', header=False)
                            else:
                                df = pd.DataFrame([row])
                                df.to_csv('malware_apis.csv')
                        else:
                            if os.path.isfile('benign_apis.csv'):
                                df = pd.DataFrame([row])
                                df.to_csv('benign_apis.csv', mode='a', header=False)
                            else:
                                df = pd.DataFrame([row])
                                df.to_csv('benign_apis.csv')
                    else:
                        return [row]
                    row = dict.fromkeys(api_features, 0)
                    count += 1
                    if count % 100 == 0:
                        print(f'finished labelling {str(count)} files')

                    current_hash = md5_hash
                    print(current_hash + ' ' + str(count))
                    if LIMIT:
                        if count >= MAX_COLLECT:
                            return

                api_calls = extract_api_calls(file_dest)
                for api_call in api_calls:
                    if api_call in api_features:
                        row[api_call] += 1


def get_features():
    # print('extracting permissions features...')
    # extract_permission_features(root_dir='../samples/malware_samples/decompiled', malware=True)
    # extract_permission_features(root_dir='../samples/benign_samples/decompiled', malware=False)
    # print('finished extracting permissions')

    # print('extracting permissions features...')
    # print('extracting malware permissions features...')
    # malware_perms = extract_perms_features(root_dir='../samples/malware_samples/decompiled', feature_count=300)
    # print('extracting benign permissions features...')
    # benign_perms = extract_perms_features(root_dir='../samples/benign_samples/decompiled', feature_count=50,
    #                                           exclude=malware_perms)
    # perm_features = list(set(malware_perms + benign_perms))
    # benign_perms = "\n".join(benign_perms)
    # with open('benign_perms.txt', 'w') as file:
    #     file.write(benign_perms)
    # malware_perms = "\n".join(malware_perms)
    # with open('malware_perms.txt', 'w') as file:
    #     file.write(malware_perms)
    # perm_features_str = "\n".join(perm_features)
    # print('\n\n\n')
    # print('perm features')
    # print('----------------------------------')
    # print(perm_features_str)
    # with open('perm_features.txt', 'w') as file:
    #     file.write(perm_features_str)
    # print('finished extracting perm features...')
    # print('extracting malware perm data...')
    # labeled_perm_data(root_dir='../samples/malware_samples/decompiled', perm_features=perm_features, malware=True)
    # print('extracting benign perm data...')
    # labeled_perm_data(root_dir='../samples/benign_samples/decompiled', perm_features=perm_features, malware=False)
    # print('finished extracting perm data...')


    print('extracting intent features...')
    print('extracting malware intent features...')
    malware_intents = extract_intents_features(root_dir='../samples/malware_samples/decompiled', feature_count=300)

    print('extracting benign intent features...')
    benign_intents = extract_intents_features(root_dir='../samples/benign_samples/decompiled', feature_count=50,
                                              exclude=malware_intents)
    intent_features = list(set(malware_intents + benign_intents))
    benign_intents = "\n".join(benign_intents)
    with open('benign_intents.txt', 'w') as file:
        file.write(benign_intents)
    malware_intents = "\n".join(malware_intents)
    with open('malware_intents.txt', 'w') as file:
        file.write(malware_intents)
    intent_features_str = "\n".join(intent_features)
    print('\n\n\n')
    print('intent features')
    print('----------------------------------')
    print(intent_features_str)
    with open('intent_features.txt', 'w') as file:
        file.write(intent_features_str)
    print('finished extracting intent features...')
    # with open('intent_features.txt', 'r') as file:
    #     intent_features = file.read().splitlines()
    print('extracting malware intent data...')
    labeled_intent_data(root_dir='../samples/malware_samples/decompiled', intent_features=intent_features, malware=True)
    print('extracting benign intent data...')
    labeled_intent_data(root_dir='../samples/benign_samples/decompiled', intent_features=intent_features, malware=False)
    print('finished extracting intent data...')







    #
    # print('extracting api features...')
    # print('extracting malware api features...')
    # malware_apis = extract_api_features(root_dir='../samples/malware_samples/decompiled', feature_count=300)
    # print('extracting benign api features...')
    # benign_apis = extract_api_features(root_dir='../samples/benign_samples/decompiled', feature_count=50,
    #                                           exclude=malware_apis)
    # api_features = list(set(malware_apis + benign_apis))
    # benign_apis = "\n".join(benign_apis)
    # with open('benign_apis.txt', 'w') as file:
    #     file.write(benign_apis)
    # malware_apis = "\n".join(malware_apis)
    # with open('malware_apis.txt', 'w') as file:
    #     file.write(malware_apis)
    # api_features_str = "\n".join(api_features)
    # print('\n\n\n')
    # print('api features')
    # print('----------------------------------')
    # print(api_features_str)
    # with open('api_features.txt', 'w') as file:
    #     file.write(api_features_str)
    # print('finished extracting api features...')
    #
    # print('extracting malware api data...')
    # labeled_api_data(root_dir='../samples/malware_samples/decompiled', api_features=api_features, malware=True)
    # print('extracting benign api data...')
    # labeled_api_data(root_dir='../samples/benign_samples/decompiled', api_features=api_features, malware=False)
    # print(f'finished extracting api data...')
    # print('Done!')

# get_features()
