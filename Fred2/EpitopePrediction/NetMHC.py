# This code is part of the Fred2 distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
__author__ = 'szolek', 'walzer'
'DEPRECATED'
#
# import re
# import logging
# import subprocess
#
# from tempfile import NamedTemporaryFile
# from itertools import groupby
#
# from Core.Base import MetadataLogger, AASequence, Score
# from Core.Allele import Allele
# import Core
# import IO
#
#
# class NetMHC(MetadataLogger):
#     def __init__(self, netmhc=None, netpan=None):
#         #N.B. Allele max number is 80!
#         MetadataLogger.__init__(self)
#         self.netmhc_path = netmhc
#         self.netpan_path = netpan
#         self.mhcheader = ["pos", "peptide", "logscore", "affinity(nM)", "Bind Level", "Protein Name", "Allele"]
#         self.panheader = ["pos", "Allele", "peptide", "Protein Name", "logscore", "affinity(nM)", "%Rank", "BindLevel"]
#
#         self.mhcalleles = ['A0101', 'A0201', 'A0202', 'A0203', 'A0204', 'A0206', 'A0211', 'A0212', 'A0216', 'A0219',
#                            'A0301', 'A1101', 'A2301', 'A2402', 'A2403', 'A2601', 'A2602', 'A2902', 'A3001', 'A3002',
#                            'A3101', 'A3301', 'A6801', 'A6802', 'A6901', 'B0702', 'B0801', 'B0802', 'B1501', 'B1801',
#                            'B2705', 'B3501', 'B3901', 'B4001', 'B4002', 'B4402', 'B4403', 'B4501', 'B5101', 'B5301',
#                            'B5401', 'B5701', 'B5801']
#
#         self.panalleles = ['BoLA-AW10', 'BoLA-D18.4', 'BoLA-HD6', 'BoLA-JSP.1', 'BoLA-N:00101', 'BoLA-N:00102',
#                             'BoLA-N:00103', 'BoLA-N:00201', 'BoLA-N:00301', 'BoLA-N:00401', 'BoLA-N:00402', 'BoLA-N:00501',
#                             'BoLA-N:00601', 'BoLA-N:00602', 'BoLA-N:00801', 'BoLA-N:00802', 'BoLA-N:00901', 'BoLA-N:00902',
#                             'BoLA-N:01001', 'BoLA-N:01101', 'BoLA-N:01201', 'BoLA-N:01301', 'BoLA-N:01302', 'BoLA-N:01401',
#                             'BoLA-N:01402', 'BoLA-N:01501', 'BoLA-N:01502', 'BoLA-N:01601', 'BoLA-N:01602', 'BoLA-N:01701',
#                             'BoLA-N:01702', 'BoLA-N:01801', 'BoLA-N:01802', 'BoLA-N:01901', 'BoLA-N:02001', 'BoLA-N:02101',
#                             'BoLA-N:02201', 'BoLA-N:02301', 'BoLA-N:02401', 'BoLA-N:02402', 'BoLA-N:02501', 'BoLA-N:02601',
#                             'BoLA-N:02602', 'BoLA-N:02701', 'BoLA-N:02702', 'BoLA-N:02801', 'BoLA-N:02901', 'BoLA-N:03001',
#                             'BoLA-N:03101', 'BoLA-N:03401', 'BoLA-N:03501', 'BoLA-N:03601', 'BoLA-N:03701', 'BoLA-N:03801',
#                             'BoLA-N:03901', 'BoLA-N:04001', 'BoLA-N:04101', 'BoLA-N:04201', 'BoLA-N:04301', 'BoLA-N:04401',
#                             'BoLA-N:04501', 'BoLA-N:04601', 'BoLA-N:04701', 'BoLA-N:04801', 'BoLA-N:04901', 'BoLA-N:05001',
#                             'BoLA-N:05101', 'BoLA-N:05201', 'BoLA-N:05301', 'BoLA-N:05401', 'BoLA-N:05501', 'BoLA-N:05601',
#                             'BoLA-NC1:00101', 'BoLA-NC1:00201', 'BoLA-NC1:00301', 'BoLA-NC1:00401', 'BoLA-NC2:00101', 'BoLA-NC2:00102',
#                             'BoLA-NC3:00101', 'BoLA-NC4:00101', 'BoLA-NC4:00201', 'BoLA-T2a', 'BoLA-T2b', 'BoLA-T2c',
#                             'BoLA-T5', 'BoLA-T7', 'Gogo-B0101', 'H-2-Db', 'H-2-Dd', 'H-2-Kb',
#                             'H-2-Kd', 'H-2-Kk', 'H-2-Ld', 'HLA-A01:01', 'HLA-A01:02', 'HLA-A01:03',
#                             'HLA-A01:06', 'HLA-A01:07', 'HLA-A01:08', 'HLA-A01:09', 'HLA-A01:10', 'HLA-A01:12',
#                             'HLA-A01:13', 'HLA-A01:14', 'HLA-A01:17', 'HLA-A01:19', 'HLA-A01:20', 'HLA-A01:21',
#                             'HLA-A01:23', 'HLA-A01:24', 'HLA-A01:25', 'HLA-A01:26', 'HLA-A01:28', 'HLA-A01:29',
#                             'HLA-A01:30', 'HLA-A01:32', 'HLA-A01:33', 'HLA-A01:35', 'HLA-A01:36', 'HLA-A01:37',
#                             'HLA-A01:38', 'HLA-A01:39', 'HLA-A01:40', 'HLA-A01:41', 'HLA-A01:42', 'HLA-A01:43',
#                             'HLA-A01:44', 'HLA-A01:45', 'HLA-A01:46', 'HLA-A01:47', 'HLA-A01:48', 'HLA-A01:49',
#                             'HLA-A01:50', 'HLA-A01:51', 'HLA-A01:54', 'HLA-A01:55', 'HLA-A01:58', 'HLA-A01:59',
#                             'HLA-A01:60', 'HLA-A01:61', 'HLA-A01:62', 'HLA-A01:63', 'HLA-A01:64', 'HLA-A01:65',
#                             'HLA-A01:66', 'HLA-A02:01', 'HLA-A02:02', 'HLA-A02:03', 'HLA-A02:04', 'HLA-A02:05',
#                             'HLA-A02:06', 'HLA-A02:07', 'HLA-A02:08', 'HLA-A02:09', 'HLA-A02:10', 'HLA-A02:101',
#                             'HLA-A02:102', 'HLA-A02:103', 'HLA-A02:104', 'HLA-A02:105', 'HLA-A02:106', 'HLA-A02:107',
#                             'HLA-A02:108', 'HLA-A02:109', 'HLA-A02:11', 'HLA-A02:110', 'HLA-A02:111', 'HLA-A02:112',
#                             'HLA-A02:114', 'HLA-A02:115', 'HLA-A02:116', 'HLA-A02:117', 'HLA-A02:118', 'HLA-A02:119',
#                             'HLA-A02:12', 'HLA-A02:120', 'HLA-A02:121', 'HLA-A02:122', 'HLA-A02:123', 'HLA-A02:124',
#                             'HLA-A02:126', 'HLA-A02:127', 'HLA-A02:128', 'HLA-A02:129', 'HLA-A02:13', 'HLA-A02:130',
#                             'HLA-A02:131', 'HLA-A02:132', 'HLA-A02:133', 'HLA-A02:134', 'HLA-A02:135', 'HLA-A02:136',
#                             'HLA-A02:137', 'HLA-A02:138', 'HLA-A02:139', 'HLA-A02:14', 'HLA-A02:140', 'HLA-A02:141',
#                             'HLA-A02:142', 'HLA-A02:143', 'HLA-A02:144', 'HLA-A02:145', 'HLA-A02:146', 'HLA-A02:147',
#                             'HLA-A02:148', 'HLA-A02:149', 'HLA-A02:150', 'HLA-A02:151', 'HLA-A02:152', 'HLA-A02:153',
#                             'HLA-A02:154', 'HLA-A02:155', 'HLA-A02:156', 'HLA-A02:157', 'HLA-A02:158', 'HLA-A02:159',
#                             'HLA-A02:16', 'HLA-A02:160', 'HLA-A02:161', 'HLA-A02:162', 'HLA-A02:163', 'HLA-A02:164',
#                             'HLA-A02:165', 'HLA-A02:166', 'HLA-A02:167', 'HLA-A02:168', 'HLA-A02:169', 'HLA-A02:17',
#                             'HLA-A02:170', 'HLA-A02:171', 'HLA-A02:172', 'HLA-A02:173', 'HLA-A02:174', 'HLA-A02:175',
#                             'HLA-A02:176', 'HLA-A02:177', 'HLA-A02:178', 'HLA-A02:179', 'HLA-A02:18', 'HLA-A02:180',
#                             'HLA-A02:181', 'HLA-A02:182', 'HLA-A02:183', 'HLA-A02:184', 'HLA-A02:185', 'HLA-A02:186',
#                             'HLA-A02:187', 'HLA-A02:188', 'HLA-A02:189', 'HLA-A02:19', 'HLA-A02:190', 'HLA-A02:191',
#                             'HLA-A02:192', 'HLA-A02:193', 'HLA-A02:194', 'HLA-A02:195', 'HLA-A02:196', 'HLA-A02:197',
#                             'HLA-A02:198', 'HLA-A02:199', 'HLA-A02:20', 'HLA-A02:200', 'HLA-A02:201', 'HLA-A02:202',
#                             'HLA-A02:203', 'HLA-A02:204', 'HLA-A02:205', 'HLA-A02:206', 'HLA-A02:207', 'HLA-A02:208',
#                             'HLA-A02:209', 'HLA-A02:21', 'HLA-A02:210', 'HLA-A02:211', 'HLA-A02:212', 'HLA-A02:213',
#                             'HLA-A02:214', 'HLA-A02:215', 'HLA-A02:216', 'HLA-A02:217', 'HLA-A02:218', 'HLA-A02:219',
#                             'HLA-A02:22', 'HLA-A02:220', 'HLA-A02:221', 'HLA-A02:224', 'HLA-A02:228', 'HLA-A02:229',
#                             'HLA-A02:230', 'HLA-A02:231', 'HLA-A02:232', 'HLA-A02:233', 'HLA-A02:234', 'HLA-A02:235',
#                             'HLA-A02:236', 'HLA-A02:237', 'HLA-A02:238', 'HLA-A02:239', 'HLA-A02:24', 'HLA-A02:240',
#                             'HLA-A02:241', 'HLA-A02:242', 'HLA-A02:243', 'HLA-A02:244', 'HLA-A02:245', 'HLA-A02:246',
#                             'HLA-A02:247', 'HLA-A02:248', 'HLA-A02:249', 'HLA-A02:25', 'HLA-A02:251', 'HLA-A02:252',
#                             'HLA-A02:253', 'HLA-A02:254', 'HLA-A02:255', 'HLA-A02:256', 'HLA-A02:257', 'HLA-A02:258',
#                             'HLA-A02:259', 'HLA-A02:26', 'HLA-A02:260', 'HLA-A02:261', 'HLA-A02:262', 'HLA-A02:263',
#                             'HLA-A02:264', 'HLA-A02:265', 'HLA-A02:266', 'HLA-A02:27', 'HLA-A02:28', 'HLA-A02:29',
#                             'HLA-A02:30', 'HLA-A02:31', 'HLA-A02:33', 'HLA-A02:34', 'HLA-A02:35', 'HLA-A02:36',
#                             'HLA-A02:37', 'HLA-A02:38', 'HLA-A02:39', 'HLA-A02:40', 'HLA-A02:41', 'HLA-A02:42',
#                             'HLA-A02:44', 'HLA-A02:45', 'HLA-A02:46', 'HLA-A02:47', 'HLA-A02:48', 'HLA-A02:49',
#                             'HLA-A02:50', 'HLA-A02:51', 'HLA-A02:52', 'HLA-A02:54', 'HLA-A02:55', 'HLA-A02:56',
#                             'HLA-A02:57', 'HLA-A02:58', 'HLA-A02:59', 'HLA-A02:60', 'HLA-A02:61', 'HLA-A02:62',
#                             'HLA-A02:63', 'HLA-A02:64', 'HLA-A02:65', 'HLA-A02:66', 'HLA-A02:67', 'HLA-A02:68',
#                             'HLA-A02:69', 'HLA-A02:70', 'HLA-A02:71', 'HLA-A02:72', 'HLA-A02:73', 'HLA-A02:74',
#                             'HLA-A02:75', 'HLA-A02:76', 'HLA-A02:77', 'HLA-A02:78', 'HLA-A02:79', 'HLA-A02:80',
#                             'HLA-A02:81', 'HLA-A02:84', 'HLA-A02:85', 'HLA-A02:86', 'HLA-A02:87', 'HLA-A02:89',
#                             'HLA-A02:90', 'HLA-A02:91', 'HLA-A02:92', 'HLA-A02:93', 'HLA-A02:95', 'HLA-A02:96',
#                             'HLA-A02:97', 'HLA-A02:99', 'HLA-A03:01', 'HLA-A03:02', 'HLA-A03:04', 'HLA-A03:05',
#                             'HLA-A03:06', 'HLA-A03:07', 'HLA-A03:08', 'HLA-A03:09', 'HLA-A03:10', 'HLA-A03:12',
#                             'HLA-A03:13', 'HLA-A03:14', 'HLA-A03:15', 'HLA-A03:16', 'HLA-A03:17', 'HLA-A03:18',
#                             'HLA-A03:19', 'HLA-A03:20', 'HLA-A03:22', 'HLA-A03:23', 'HLA-A03:24', 'HLA-A03:25',
#                             'HLA-A03:26', 'HLA-A03:27', 'HLA-A03:28', 'HLA-A03:29', 'HLA-A03:30', 'HLA-A03:31',
#                             'HLA-A03:32', 'HLA-A03:33', 'HLA-A03:34', 'HLA-A03:35', 'HLA-A03:37', 'HLA-A03:38',
#                             'HLA-A03:39', 'HLA-A03:40', 'HLA-A03:41', 'HLA-A03:42', 'HLA-A03:43', 'HLA-A03:44',
#                             'HLA-A03:45', 'HLA-A03:46', 'HLA-A03:47', 'HLA-A03:48', 'HLA-A03:49', 'HLA-A03:50',
#                             'HLA-A03:51', 'HLA-A03:52', 'HLA-A03:53', 'HLA-A03:54', 'HLA-A03:55', 'HLA-A03:56',
#                             'HLA-A03:57', 'HLA-A03:58', 'HLA-A03:59', 'HLA-A03:60', 'HLA-A03:61', 'HLA-A03:62',
#                             'HLA-A03:63', 'HLA-A03:64', 'HLA-A03:65', 'HLA-A03:66', 'HLA-A03:67', 'HLA-A03:70',
#                             'HLA-A03:71', 'HLA-A03:72', 'HLA-A03:73', 'HLA-A03:74', 'HLA-A03:75', 'HLA-A03:76',
#                             'HLA-A03:77', 'HLA-A03:78', 'HLA-A03:79', 'HLA-A03:80', 'HLA-A03:81', 'HLA-A03:82',
#                             'HLA-A11:01', 'HLA-A11:02', 'HLA-A11:03', 'HLA-A11:04', 'HLA-A11:05', 'HLA-A11:06',
#                             'HLA-A11:07', 'HLA-A11:08', 'HLA-A11:09', 'HLA-A11:10', 'HLA-A11:11', 'HLA-A11:12',
#                             'HLA-A11:13', 'HLA-A11:14', 'HLA-A11:15', 'HLA-A11:16', 'HLA-A11:17', 'HLA-A11:18',
#                             'HLA-A11:19', 'HLA-A11:20', 'HLA-A11:22', 'HLA-A11:23', 'HLA-A11:24', 'HLA-A11:25',
#                             'HLA-A11:26', 'HLA-A11:27', 'HLA-A11:29', 'HLA-A11:30', 'HLA-A11:31', 'HLA-A11:32',
#                             'HLA-A11:33', 'HLA-A11:34', 'HLA-A11:35', 'HLA-A11:36', 'HLA-A11:37', 'HLA-A11:38',
#                             'HLA-A11:39', 'HLA-A11:40', 'HLA-A11:41', 'HLA-A11:42', 'HLA-A11:43', 'HLA-A11:44',
#                             'HLA-A11:45', 'HLA-A11:46', 'HLA-A11:47', 'HLA-A11:48', 'HLA-A11:49', 'HLA-A11:51',
#                             'HLA-A11:53', 'HLA-A11:54', 'HLA-A11:55', 'HLA-A11:56', 'HLA-A11:57', 'HLA-A11:58',
#                             'HLA-A11:59', 'HLA-A11:60', 'HLA-A11:61', 'HLA-A11:62', 'HLA-A11:63', 'HLA-A11:64',
#                             'HLA-A23:01', 'HLA-A23:02', 'HLA-A23:03', 'HLA-A23:04', 'HLA-A23:05', 'HLA-A23:06',
#                             'HLA-A23:09', 'HLA-A23:10', 'HLA-A23:12', 'HLA-A23:13', 'HLA-A23:14', 'HLA-A23:15',
#                             'HLA-A23:16', 'HLA-A23:17', 'HLA-A23:18', 'HLA-A23:20', 'HLA-A23:21', 'HLA-A23:22',
#                             'HLA-A23:23', 'HLA-A23:24', 'HLA-A23:25', 'HLA-A23:26', 'HLA-A24:02', 'HLA-A24:03',
#                             'HLA-A24:04', 'HLA-A24:05', 'HLA-A24:06', 'HLA-A24:07', 'HLA-A24:08', 'HLA-A24:10',
#                             'HLA-A24:100', 'HLA-A24:101', 'HLA-A24:102', 'HLA-A24:103', 'HLA-A24:104', 'HLA-A24:105',
#                             'HLA-A24:106', 'HLA-A24:107', 'HLA-A24:108', 'HLA-A24:109', 'HLA-A24:110', 'HLA-A24:111',
#                             'HLA-A24:112', 'HLA-A24:113', 'HLA-A24:114', 'HLA-A24:115', 'HLA-A24:116', 'HLA-A24:117',
#                             'HLA-A24:118', 'HLA-A24:119', 'HLA-A24:120', 'HLA-A24:121', 'HLA-A24:122', 'HLA-A24:123',
#                             'HLA-A24:124', 'HLA-A24:125', 'HLA-A24:126', 'HLA-A24:127', 'HLA-A24:128', 'HLA-A24:129',
#                             'HLA-A24:13', 'HLA-A24:130', 'HLA-A24:131', 'HLA-A24:133', 'HLA-A24:134', 'HLA-A24:135',
#                             'HLA-A24:136', 'HLA-A24:137', 'HLA-A24:138', 'HLA-A24:139', 'HLA-A24:14', 'HLA-A24:140',
#                             'HLA-A24:141', 'HLA-A24:142', 'HLA-A24:143', 'HLA-A24:144', 'HLA-A24:15', 'HLA-A24:17',
#                             'HLA-A24:18', 'HLA-A24:19', 'HLA-A24:20', 'HLA-A24:21', 'HLA-A24:22', 'HLA-A24:23',
#                             'HLA-A24:24', 'HLA-A24:25', 'HLA-A24:26', 'HLA-A24:27', 'HLA-A24:28', 'HLA-A24:29',
#                             'HLA-A24:30', 'HLA-A24:31', 'HLA-A24:32', 'HLA-A24:33', 'HLA-A24:34', 'HLA-A24:35',
#                             'HLA-A24:37', 'HLA-A24:38', 'HLA-A24:39', 'HLA-A24:41', 'HLA-A24:42', 'HLA-A24:43',
#                             'HLA-A24:44', 'HLA-A24:46', 'HLA-A24:47', 'HLA-A24:49', 'HLA-A24:50', 'HLA-A24:51',
#                             'HLA-A24:52', 'HLA-A24:53', 'HLA-A24:54', 'HLA-A24:55', 'HLA-A24:56', 'HLA-A24:57',
#                             'HLA-A24:58', 'HLA-A24:59', 'HLA-A24:61', 'HLA-A24:62', 'HLA-A24:63', 'HLA-A24:64',
#                             'HLA-A24:66', 'HLA-A24:67', 'HLA-A24:68', 'HLA-A24:69', 'HLA-A24:70', 'HLA-A24:71',
#                             'HLA-A24:72', 'HLA-A24:73', 'HLA-A24:74', 'HLA-A24:75', 'HLA-A24:76', 'HLA-A24:77',
#                             'HLA-A24:78', 'HLA-A24:79', 'HLA-A24:80', 'HLA-A24:81', 'HLA-A24:82', 'HLA-A24:85',
#                             'HLA-A24:87', 'HLA-A24:88', 'HLA-A24:89', 'HLA-A24:91', 'HLA-A24:92', 'HLA-A24:93',
#                             'HLA-A24:94', 'HLA-A24:95', 'HLA-A24:96', 'HLA-A24:97', 'HLA-A24:98', 'HLA-A24:99',
#                             'HLA-A25:01', 'HLA-A25:02', 'HLA-A25:03', 'HLA-A25:04', 'HLA-A25:05', 'HLA-A25:06',
#                             'HLA-A25:07', 'HLA-A25:08', 'HLA-A25:09', 'HLA-A25:10', 'HLA-A25:11', 'HLA-A25:13',
#                             'HLA-A26:01', 'HLA-A26:02', 'HLA-A26:03', 'HLA-A26:04', 'HLA-A26:05', 'HLA-A26:06',
#                             'HLA-A26:07', 'HLA-A26:08', 'HLA-A26:09', 'HLA-A26:10', 'HLA-A26:12', 'HLA-A26:13',
#                             'HLA-A26:14', 'HLA-A26:15', 'HLA-A26:16', 'HLA-A26:17', 'HLA-A26:18', 'HLA-A26:19',
#                             'HLA-A26:20', 'HLA-A26:21', 'HLA-A26:22', 'HLA-A26:23', 'HLA-A26:24', 'HLA-A26:26',
#                             'HLA-A26:27', 'HLA-A26:28', 'HLA-A26:29', 'HLA-A26:30', 'HLA-A26:31', 'HLA-A26:32',
#                             'HLA-A26:33', 'HLA-A26:34', 'HLA-A26:35', 'HLA-A26:36', 'HLA-A26:37', 'HLA-A26:38',
#                             'HLA-A26:39', 'HLA-A26:40', 'HLA-A26:41', 'HLA-A26:42', 'HLA-A26:43', 'HLA-A26:45',
#                             'HLA-A26:46', 'HLA-A26:47', 'HLA-A26:48', 'HLA-A26:49', 'HLA-A26:50', 'HLA-A29:01',
#                             'HLA-A29:02', 'HLA-A29:03', 'HLA-A29:04', 'HLA-A29:05', 'HLA-A29:06', 'HLA-A29:07',
#                             'HLA-A29:09', 'HLA-A29:10', 'HLA-A29:11', 'HLA-A29:12', 'HLA-A29:13', 'HLA-A29:14',
#                             'HLA-A29:15', 'HLA-A29:16', 'HLA-A29:17', 'HLA-A29:18', 'HLA-A29:19', 'HLA-A29:20',
#                             'HLA-A29:21', 'HLA-A29:22', 'HLA-A30:01', 'HLA-A30:02', 'HLA-A30:03', 'HLA-A30:04',
#                             'HLA-A30:06', 'HLA-A30:07', 'HLA-A30:08', 'HLA-A30:09', 'HLA-A30:10', 'HLA-A30:11',
#                             'HLA-A30:12', 'HLA-A30:13', 'HLA-A30:15', 'HLA-A30:16', 'HLA-A30:17', 'HLA-A30:18',
#                             'HLA-A30:19', 'HLA-A30:20', 'HLA-A30:22', 'HLA-A30:23', 'HLA-A30:24', 'HLA-A30:25',
#                             'HLA-A30:26', 'HLA-A30:28', 'HLA-A30:29', 'HLA-A30:30', 'HLA-A30:31', 'HLA-A30:32',
#                             'HLA-A30:33', 'HLA-A30:34', 'HLA-A30:35', 'HLA-A30:36', 'HLA-A30:37', 'HLA-A30:38',
#                             'HLA-A30:39', 'HLA-A30:40', 'HLA-A30:41', 'HLA-A31:01', 'HLA-A31:02', 'HLA-A31:03',
#                             'HLA-A31:04', 'HLA-A31:05', 'HLA-A31:06', 'HLA-A31:07', 'HLA-A31:08', 'HLA-A31:09',
#                             'HLA-A31:10', 'HLA-A31:11', 'HLA-A31:12', 'HLA-A31:13', 'HLA-A31:15', 'HLA-A31:16',
#                             'HLA-A31:17', 'HLA-A31:18', 'HLA-A31:19', 'HLA-A31:20', 'HLA-A31:21', 'HLA-A31:22',
#                             'HLA-A31:23', 'HLA-A31:24', 'HLA-A31:25', 'HLA-A31:26', 'HLA-A31:27', 'HLA-A31:28',
#                             'HLA-A31:29', 'HLA-A31:30', 'HLA-A31:31', 'HLA-A31:32', 'HLA-A31:33', 'HLA-A31:34',
#                             'HLA-A31:35', 'HLA-A31:36', 'HLA-A31:37', 'HLA-A32:01', 'HLA-A32:02', 'HLA-A32:03',
#                             'HLA-A32:04', 'HLA-A32:05', 'HLA-A32:06', 'HLA-A32:07', 'HLA-A32:08', 'HLA-A32:09',
#                             'HLA-A32:10', 'HLA-A32:12', 'HLA-A32:13', 'HLA-A32:14', 'HLA-A32:15', 'HLA-A32:16',
#                             'HLA-A32:17', 'HLA-A32:18', 'HLA-A32:20', 'HLA-A32:21', 'HLA-A32:22', 'HLA-A32:23',
#                             'HLA-A32:24', 'HLA-A32:25', 'HLA-A33:01', 'HLA-A33:03', 'HLA-A33:04', 'HLA-A33:05',
#                             'HLA-A33:06', 'HLA-A33:07', 'HLA-A33:08', 'HLA-A33:09', 'HLA-A33:10', 'HLA-A33:11',
#                             'HLA-A33:12', 'HLA-A33:13', 'HLA-A33:14', 'HLA-A33:15', 'HLA-A33:16', 'HLA-A33:17',
#                             'HLA-A33:18', 'HLA-A33:19', 'HLA-A33:20', 'HLA-A33:21', 'HLA-A33:22', 'HLA-A33:23',
#                             'HLA-A33:24', 'HLA-A33:25', 'HLA-A33:26', 'HLA-A33:27', 'HLA-A33:28', 'HLA-A33:29',
#                             'HLA-A33:30', 'HLA-A33:31', 'HLA-A34:01', 'HLA-A34:02', 'HLA-A34:03', 'HLA-A34:04',
#                             'HLA-A34:05', 'HLA-A34:06', 'HLA-A34:07', 'HLA-A34:08', 'HLA-A36:01', 'HLA-A36:02',
#                             'HLA-A36:03', 'HLA-A36:04', 'HLA-A36:05', 'HLA-A43:01', 'HLA-A66:01', 'HLA-A66:02',
#                             'HLA-A66:03', 'HLA-A66:04', 'HLA-A66:05', 'HLA-A66:06', 'HLA-A66:07', 'HLA-A66:08',
#                             'HLA-A66:09', 'HLA-A66:10', 'HLA-A66:11', 'HLA-A66:12', 'HLA-A66:13', 'HLA-A66:14',
#                             'HLA-A66:15', 'HLA-A68:01', 'HLA-A68:02', 'HLA-A68:03', 'HLA-A68:04', 'HLA-A68:05',
#                             'HLA-A68:06', 'HLA-A68:07', 'HLA-A68:08', 'HLA-A68:09', 'HLA-A68:10', 'HLA-A68:12',
#                             'HLA-A68:13', 'HLA-A68:14', 'HLA-A68:15', 'HLA-A68:16', 'HLA-A68:17', 'HLA-A68:19',
#                             'HLA-A68:20', 'HLA-A68:21', 'HLA-A68:22', 'HLA-A68:23', 'HLA-A68:24', 'HLA-A68:25',
#                             'HLA-A68:26', 'HLA-A68:27', 'HLA-A68:28', 'HLA-A68:29', 'HLA-A68:30', 'HLA-A68:31',
#                             'HLA-A68:32', 'HLA-A68:33', 'HLA-A68:34', 'HLA-A68:35', 'HLA-A68:36', 'HLA-A68:37',
#                             'HLA-A68:38', 'HLA-A68:39', 'HLA-A68:40', 'HLA-A68:41', 'HLA-A68:42', 'HLA-A68:43',
#                             'HLA-A68:44', 'HLA-A68:45', 'HLA-A68:46', 'HLA-A68:47', 'HLA-A68:48', 'HLA-A68:50',
#                             'HLA-A68:51', 'HLA-A68:52', 'HLA-A68:53', 'HLA-A68:54', 'HLA-A69:01', 'HLA-A74:01',
#                             'HLA-A74:02', 'HLA-A74:03', 'HLA-A74:04', 'HLA-A74:05', 'HLA-A74:06', 'HLA-A74:07',
#                             'HLA-A74:08', 'HLA-A74:09', 'HLA-A74:10', 'HLA-A74:11', 'HLA-A74:13', 'HLA-A80:01',
#                             'HLA-A80:02', 'HLA-B07:02', 'HLA-B07:03', 'HLA-B07:04', 'HLA-B07:05', 'HLA-B07:06',
#                             'HLA-B07:07', 'HLA-B07:08', 'HLA-B07:09', 'HLA-B07:10', 'HLA-B07:100', 'HLA-B07:101',
#                             'HLA-B07:102', 'HLA-B07:103', 'HLA-B07:104', 'HLA-B07:105', 'HLA-B07:106', 'HLA-B07:107',
#                             'HLA-B07:108', 'HLA-B07:109', 'HLA-B07:11', 'HLA-B07:110', 'HLA-B07:112', 'HLA-B07:113',
#                             'HLA-B07:114', 'HLA-B07:115', 'HLA-B07:12', 'HLA-B07:13', 'HLA-B07:14', 'HLA-B07:15',
#                             'HLA-B07:16', 'HLA-B07:17', 'HLA-B07:18', 'HLA-B07:19', 'HLA-B07:20', 'HLA-B07:21',
#                             'HLA-B07:22', 'HLA-B07:23', 'HLA-B07:24', 'HLA-B07:25', 'HLA-B07:26', 'HLA-B07:27',
#                             'HLA-B07:28', 'HLA-B07:29', 'HLA-B07:30', 'HLA-B07:31', 'HLA-B07:32', 'HLA-B07:33',
#                             'HLA-B07:34', 'HLA-B07:35', 'HLA-B07:36', 'HLA-B07:37', 'HLA-B07:38', 'HLA-B07:39',
#                             'HLA-B07:40', 'HLA-B07:41', 'HLA-B07:42', 'HLA-B07:43', 'HLA-B07:44', 'HLA-B07:45',
#                             'HLA-B07:46', 'HLA-B07:47', 'HLA-B07:48', 'HLA-B07:50', 'HLA-B07:51', 'HLA-B07:52',
#                             'HLA-B07:53', 'HLA-B07:54', 'HLA-B07:55', 'HLA-B07:56', 'HLA-B07:57', 'HLA-B07:58',
#                             'HLA-B07:59', 'HLA-B07:60', 'HLA-B07:61', 'HLA-B07:62', 'HLA-B07:63', 'HLA-B07:64',
#                             'HLA-B07:65', 'HLA-B07:66', 'HLA-B07:68', 'HLA-B07:69', 'HLA-B07:70', 'HLA-B07:71',
#                             'HLA-B07:72', 'HLA-B07:73', 'HLA-B07:74', 'HLA-B07:75', 'HLA-B07:76', 'HLA-B07:77',
#                             'HLA-B07:78', 'HLA-B07:79', 'HLA-B07:80', 'HLA-B07:81', 'HLA-B07:82', 'HLA-B07:83',
#                             'HLA-B07:84', 'HLA-B07:85', 'HLA-B07:86', 'HLA-B07:87', 'HLA-B07:88', 'HLA-B07:89',
#                             'HLA-B07:90', 'HLA-B07:91', 'HLA-B07:92', 'HLA-B07:93', 'HLA-B07:94', 'HLA-B07:95',
#                             'HLA-B07:96', 'HLA-B07:97', 'HLA-B07:98', 'HLA-B07:99', 'HLA-B08:01', 'HLA-B08:02',
#                             'HLA-B08:03', 'HLA-B08:04', 'HLA-B08:05', 'HLA-B08:07', 'HLA-B08:09', 'HLA-B08:10',
#                             'HLA-B08:11', 'HLA-B08:12', 'HLA-B08:13', 'HLA-B08:14', 'HLA-B08:15', 'HLA-B08:16',
#                             'HLA-B08:17', 'HLA-B08:18', 'HLA-B08:20', 'HLA-B08:21', 'HLA-B08:22', 'HLA-B08:23',
#                             'HLA-B08:24', 'HLA-B08:25', 'HLA-B08:26', 'HLA-B08:27', 'HLA-B08:28', 'HLA-B08:29',
#                             'HLA-B08:31', 'HLA-B08:32', 'HLA-B08:33', 'HLA-B08:34', 'HLA-B08:35', 'HLA-B08:36',
#                             'HLA-B08:37', 'HLA-B08:38', 'HLA-B08:39', 'HLA-B08:40', 'HLA-B08:41', 'HLA-B08:42',
#                             'HLA-B08:43', 'HLA-B08:44', 'HLA-B08:45', 'HLA-B08:46', 'HLA-B08:47', 'HLA-B08:48',
#                             'HLA-B08:49', 'HLA-B08:50', 'HLA-B08:51', 'HLA-B08:52', 'HLA-B08:53', 'HLA-B08:54',
#                             'HLA-B08:55', 'HLA-B08:56', 'HLA-B08:57', 'HLA-B08:58', 'HLA-B08:59', 'HLA-B08:60',
#                             'HLA-B08:61', 'HLA-B08:62', 'HLA-B13:01', 'HLA-B13:02', 'HLA-B13:03', 'HLA-B13:04',
#                             'HLA-B13:06', 'HLA-B13:09', 'HLA-B13:10', 'HLA-B13:11', 'HLA-B13:12', 'HLA-B13:13',
#                             'HLA-B13:14', 'HLA-B13:15', 'HLA-B13:16', 'HLA-B13:17', 'HLA-B13:18', 'HLA-B13:19',
#                             'HLA-B13:20', 'HLA-B13:21', 'HLA-B13:22', 'HLA-B13:23', 'HLA-B13:25', 'HLA-B13:26',
#                             'HLA-B13:27', 'HLA-B13:28', 'HLA-B13:29', 'HLA-B13:30', 'HLA-B13:31', 'HLA-B13:32',
#                             'HLA-B13:33', 'HLA-B13:34', 'HLA-B13:35', 'HLA-B13:36', 'HLA-B13:37', 'HLA-B13:38',
#                             'HLA-B13:39', 'HLA-B14:01', 'HLA-B14:02', 'HLA-B14:03', 'HLA-B14:04', 'HLA-B14:05',
#                             'HLA-B14:06', 'HLA-B14:08', 'HLA-B14:09', 'HLA-B14:10', 'HLA-B14:11', 'HLA-B14:12',
#                             'HLA-B14:13', 'HLA-B14:14', 'HLA-B14:15', 'HLA-B14:16', 'HLA-B14:17', 'HLA-B14:18',
#                             'HLA-B15:01', 'HLA-B15:02', 'HLA-B15:03', 'HLA-B15:04', 'HLA-B15:05', 'HLA-B15:06',
#                             'HLA-B15:07', 'HLA-B15:08', 'HLA-B15:09', 'HLA-B15:10', 'HLA-B15:101', 'HLA-B15:102',
#                             'HLA-B15:103', 'HLA-B15:104', 'HLA-B15:105', 'HLA-B15:106', 'HLA-B15:107', 'HLA-B15:108',
#                             'HLA-B15:109', 'HLA-B15:11', 'HLA-B15:110', 'HLA-B15:112', 'HLA-B15:113', 'HLA-B15:114',
#                             'HLA-B15:115', 'HLA-B15:116', 'HLA-B15:117', 'HLA-B15:118', 'HLA-B15:119', 'HLA-B15:12',
#                             'HLA-B15:120', 'HLA-B15:121', 'HLA-B15:122', 'HLA-B15:123', 'HLA-B15:124', 'HLA-B15:125',
#                             'HLA-B15:126', 'HLA-B15:127', 'HLA-B15:128', 'HLA-B15:129', 'HLA-B15:13', 'HLA-B15:131',
#                             'HLA-B15:132', 'HLA-B15:133', 'HLA-B15:134', 'HLA-B15:135', 'HLA-B15:136', 'HLA-B15:137',
#                             'HLA-B15:138', 'HLA-B15:139', 'HLA-B15:14', 'HLA-B15:140', 'HLA-B15:141', 'HLA-B15:142',
#                             'HLA-B15:143', 'HLA-B15:144', 'HLA-B15:145', 'HLA-B15:146', 'HLA-B15:147', 'HLA-B15:148',
#                             'HLA-B15:15', 'HLA-B15:150', 'HLA-B15:151', 'HLA-B15:152', 'HLA-B15:153', 'HLA-B15:154',
#                             'HLA-B15:155', 'HLA-B15:156', 'HLA-B15:157', 'HLA-B15:158', 'HLA-B15:159', 'HLA-B15:16',
#                             'HLA-B15:160', 'HLA-B15:161', 'HLA-B15:162', 'HLA-B15:163', 'HLA-B15:164', 'HLA-B15:165',
#                             'HLA-B15:166', 'HLA-B15:167', 'HLA-B15:168', 'HLA-B15:169', 'HLA-B15:17', 'HLA-B15:170',
#                             'HLA-B15:171', 'HLA-B15:172', 'HLA-B15:173', 'HLA-B15:174', 'HLA-B15:175', 'HLA-B15:176',
#                             'HLA-B15:177', 'HLA-B15:178', 'HLA-B15:179', 'HLA-B15:18', 'HLA-B15:180', 'HLA-B15:183',
#                             'HLA-B15:184', 'HLA-B15:185', 'HLA-B15:186', 'HLA-B15:187', 'HLA-B15:188', 'HLA-B15:189',
#                             'HLA-B15:19', 'HLA-B15:191', 'HLA-B15:192', 'HLA-B15:193', 'HLA-B15:194', 'HLA-B15:195',
#                             'HLA-B15:196', 'HLA-B15:197', 'HLA-B15:198', 'HLA-B15:199', 'HLA-B15:20', 'HLA-B15:200',
#                             'HLA-B15:201', 'HLA-B15:202', 'HLA-B15:21', 'HLA-B15:23', 'HLA-B15:24', 'HLA-B15:25',
#                             'HLA-B15:27', 'HLA-B15:28', 'HLA-B15:29', 'HLA-B15:30', 'HLA-B15:31', 'HLA-B15:32',
#                             'HLA-B15:33', 'HLA-B15:34', 'HLA-B15:35', 'HLA-B15:36', 'HLA-B15:37', 'HLA-B15:38',
#                             'HLA-B15:39', 'HLA-B15:40', 'HLA-B15:42', 'HLA-B15:43', 'HLA-B15:44', 'HLA-B15:45',
#                             'HLA-B15:46', 'HLA-B15:47', 'HLA-B15:48', 'HLA-B15:49', 'HLA-B15:50', 'HLA-B15:51',
#                             'HLA-B15:52', 'HLA-B15:53', 'HLA-B15:54', 'HLA-B15:55', 'HLA-B15:56', 'HLA-B15:57',
#                             'HLA-B15:58', 'HLA-B15:60', 'HLA-B15:61', 'HLA-B15:62', 'HLA-B15:63', 'HLA-B15:64',
#                             'HLA-B15:65', 'HLA-B15:66', 'HLA-B15:67', 'HLA-B15:68', 'HLA-B15:69', 'HLA-B15:70',
#                             'HLA-B15:71', 'HLA-B15:72', 'HLA-B15:73', 'HLA-B15:74', 'HLA-B15:75', 'HLA-B15:76',
#                             'HLA-B15:77', 'HLA-B15:78', 'HLA-B15:80', 'HLA-B15:81', 'HLA-B15:82', 'HLA-B15:83',
#                             'HLA-B15:84', 'HLA-B15:85', 'HLA-B15:86', 'HLA-B15:87', 'HLA-B15:88', 'HLA-B15:89',
#                             'HLA-B15:90', 'HLA-B15:91', 'HLA-B15:92', 'HLA-B15:93', 'HLA-B15:95', 'HLA-B15:96',
#                             'HLA-B15:97', 'HLA-B15:98', 'HLA-B15:99', 'HLA-B18:01', 'HLA-B18:02', 'HLA-B18:03',
#                             'HLA-B18:04', 'HLA-B18:05', 'HLA-B18:06', 'HLA-B18:07', 'HLA-B18:08', 'HLA-B18:09',
#                             'HLA-B18:10', 'HLA-B18:11', 'HLA-B18:12', 'HLA-B18:13', 'HLA-B18:14', 'HLA-B18:15',
#                             'HLA-B18:18', 'HLA-B18:19', 'HLA-B18:20', 'HLA-B18:21', 'HLA-B18:22', 'HLA-B18:24',
#                             'HLA-B18:25', 'HLA-B18:26', 'HLA-B18:27', 'HLA-B18:28', 'HLA-B18:29', 'HLA-B18:30',
#                             'HLA-B18:31', 'HLA-B18:32', 'HLA-B18:33', 'HLA-B18:34', 'HLA-B18:35', 'HLA-B18:36',
#                             'HLA-B18:37', 'HLA-B18:38', 'HLA-B18:39', 'HLA-B18:40', 'HLA-B18:41', 'HLA-B18:42',
#                             'HLA-B18:43', 'HLA-B18:44', 'HLA-B18:45', 'HLA-B18:46', 'HLA-B18:47', 'HLA-B18:48',
#                             'HLA-B18:49', 'HLA-B18:50', 'HLA-B27:01', 'HLA-B27:02', 'HLA-B27:03', 'HLA-B27:04',
#                             'HLA-B27:05', 'HLA-B27:06', 'HLA-B27:07', 'HLA-B27:08', 'HLA-B27:09', 'HLA-B27:10',
#                             'HLA-B27:11', 'HLA-B27:12', 'HLA-B27:13', 'HLA-B27:14', 'HLA-B27:15', 'HLA-B27:16',
#                             'HLA-B27:17', 'HLA-B27:18', 'HLA-B27:19', 'HLA-B27:20', 'HLA-B27:21', 'HLA-B27:23',
#                             'HLA-B27:24', 'HLA-B27:25', 'HLA-B27:26', 'HLA-B27:27', 'HLA-B27:28', 'HLA-B27:29',
#                             'HLA-B27:30', 'HLA-B27:31', 'HLA-B27:32', 'HLA-B27:33', 'HLA-B27:34', 'HLA-B27:35',
#                             'HLA-B27:36', 'HLA-B27:37', 'HLA-B27:38', 'HLA-B27:39', 'HLA-B27:40', 'HLA-B27:41',
#                             'HLA-B27:42', 'HLA-B27:43', 'HLA-B27:44', 'HLA-B27:45', 'HLA-B27:46', 'HLA-B27:47',
#                             'HLA-B27:48', 'HLA-B27:49', 'HLA-B27:50', 'HLA-B27:51', 'HLA-B27:52', 'HLA-B27:53',
#                             'HLA-B27:54', 'HLA-B27:55', 'HLA-B27:56', 'HLA-B27:57', 'HLA-B27:58', 'HLA-B27:60',
#                             'HLA-B27:61', 'HLA-B27:62', 'HLA-B27:63', 'HLA-B27:67', 'HLA-B27:68', 'HLA-B27:69',
#                             'HLA-B35:01', 'HLA-B35:02', 'HLA-B35:03', 'HLA-B35:04', 'HLA-B35:05', 'HLA-B35:06',
#                             'HLA-B35:07', 'HLA-B35:08', 'HLA-B35:09', 'HLA-B35:10', 'HLA-B35:100', 'HLA-B35:101',
#                             'HLA-B35:102', 'HLA-B35:103', 'HLA-B35:104', 'HLA-B35:105', 'HLA-B35:106', 'HLA-B35:107',
#                             'HLA-B35:108', 'HLA-B35:109', 'HLA-B35:11', 'HLA-B35:110', 'HLA-B35:111', 'HLA-B35:112',
#                             'HLA-B35:113', 'HLA-B35:114', 'HLA-B35:115', 'HLA-B35:116', 'HLA-B35:117', 'HLA-B35:118',
#                             'HLA-B35:119', 'HLA-B35:12', 'HLA-B35:120', 'HLA-B35:121', 'HLA-B35:122', 'HLA-B35:123',
#                             'HLA-B35:124', 'HLA-B35:125', 'HLA-B35:126', 'HLA-B35:127', 'HLA-B35:128', 'HLA-B35:13',
#                             'HLA-B35:131', 'HLA-B35:132', 'HLA-B35:133', 'HLA-B35:135', 'HLA-B35:136', 'HLA-B35:137',
#                             'HLA-B35:138', 'HLA-B35:139', 'HLA-B35:14', 'HLA-B35:140', 'HLA-B35:141', 'HLA-B35:142',
#                             'HLA-B35:143', 'HLA-B35:144', 'HLA-B35:15', 'HLA-B35:16', 'HLA-B35:17', 'HLA-B35:18',
#                             'HLA-B35:19', 'HLA-B35:20', 'HLA-B35:21', 'HLA-B35:22', 'HLA-B35:23', 'HLA-B35:24',
#                             'HLA-B35:25', 'HLA-B35:26', 'HLA-B35:27', 'HLA-B35:28', 'HLA-B35:29', 'HLA-B35:30',
#                             'HLA-B35:31', 'HLA-B35:32', 'HLA-B35:33', 'HLA-B35:34', 'HLA-B35:35', 'HLA-B35:36',
#                             'HLA-B35:37', 'HLA-B35:38', 'HLA-B35:39', 'HLA-B35:41', 'HLA-B35:42', 'HLA-B35:43',
#                             'HLA-B35:44', 'HLA-B35:45', 'HLA-B35:46', 'HLA-B35:47', 'HLA-B35:48', 'HLA-B35:49',
#                             'HLA-B35:50', 'HLA-B35:51', 'HLA-B35:52', 'HLA-B35:54', 'HLA-B35:55', 'HLA-B35:56',
#                             'HLA-B35:57', 'HLA-B35:58', 'HLA-B35:59', 'HLA-B35:60', 'HLA-B35:61', 'HLA-B35:62',
#                             'HLA-B35:63', 'HLA-B35:64', 'HLA-B35:66', 'HLA-B35:67', 'HLA-B35:68', 'HLA-B35:69',
#                             'HLA-B35:70', 'HLA-B35:71', 'HLA-B35:72', 'HLA-B35:74', 'HLA-B35:75', 'HLA-B35:76',
#                             'HLA-B35:77', 'HLA-B35:78', 'HLA-B35:79', 'HLA-B35:80', 'HLA-B35:81', 'HLA-B35:82',
#                             'HLA-B35:83', 'HLA-B35:84', 'HLA-B35:85', 'HLA-B35:86', 'HLA-B35:87', 'HLA-B35:88',
#                             'HLA-B35:89', 'HLA-B35:90', 'HLA-B35:91', 'HLA-B35:92', 'HLA-B35:93', 'HLA-B35:94',
#                             'HLA-B35:95', 'HLA-B35:96', 'HLA-B35:97', 'HLA-B35:98', 'HLA-B35:99', 'HLA-B37:01',
#                             'HLA-B37:02', 'HLA-B37:04', 'HLA-B37:05', 'HLA-B37:06', 'HLA-B37:07', 'HLA-B37:08',
#                             'HLA-B37:09', 'HLA-B37:10', 'HLA-B37:11', 'HLA-B37:12', 'HLA-B37:13', 'HLA-B37:14',
#                             'HLA-B37:15', 'HLA-B37:17', 'HLA-B37:18', 'HLA-B37:19', 'HLA-B37:20', 'HLA-B37:21',
#                             'HLA-B37:22', 'HLA-B37:23', 'HLA-B38:01', 'HLA-B38:02', 'HLA-B38:03', 'HLA-B38:04',
#                             'HLA-B38:05', 'HLA-B38:06', 'HLA-B38:07', 'HLA-B38:08', 'HLA-B38:09', 'HLA-B38:10',
#                             'HLA-B38:11', 'HLA-B38:12', 'HLA-B38:13', 'HLA-B38:14', 'HLA-B38:15', 'HLA-B38:16',
#                             'HLA-B38:17', 'HLA-B38:18', 'HLA-B38:19', 'HLA-B38:20', 'HLA-B38:21', 'HLA-B38:22',
#                             'HLA-B38:23', 'HLA-B39:01', 'HLA-B39:02', 'HLA-B39:03', 'HLA-B39:04', 'HLA-B39:05',
#                             'HLA-B39:06', 'HLA-B39:07', 'HLA-B39:08', 'HLA-B39:09', 'HLA-B39:10', 'HLA-B39:11',
#                             'HLA-B39:12', 'HLA-B39:13', 'HLA-B39:14', 'HLA-B39:15', 'HLA-B39:16', 'HLA-B39:17',
#                             'HLA-B39:18', 'HLA-B39:19', 'HLA-B39:20', 'HLA-B39:22', 'HLA-B39:23', 'HLA-B39:24',
#                             'HLA-B39:26', 'HLA-B39:27', 'HLA-B39:28', 'HLA-B39:29', 'HLA-B39:30', 'HLA-B39:31',
#                             'HLA-B39:32', 'HLA-B39:33', 'HLA-B39:34', 'HLA-B39:35', 'HLA-B39:36', 'HLA-B39:37',
#                             'HLA-B39:39', 'HLA-B39:41', 'HLA-B39:42', 'HLA-B39:43', 'HLA-B39:44', 'HLA-B39:45',
#                             'HLA-B39:46', 'HLA-B39:47', 'HLA-B39:48', 'HLA-B39:49', 'HLA-B39:50', 'HLA-B39:51',
#                             'HLA-B39:52', 'HLA-B39:53', 'HLA-B39:54', 'HLA-B39:55', 'HLA-B39:56', 'HLA-B39:57',
#                             'HLA-B39:58', 'HLA-B39:59', 'HLA-B39:60', 'HLA-B40:01', 'HLA-B40:02', 'HLA-B40:03',
#                             'HLA-B40:04', 'HLA-B40:05', 'HLA-B40:06', 'HLA-B40:07', 'HLA-B40:08', 'HLA-B40:09',
#                             'HLA-B40:10', 'HLA-B40:100', 'HLA-B40:101', 'HLA-B40:102', 'HLA-B40:103', 'HLA-B40:104',
#                             'HLA-B40:105', 'HLA-B40:106', 'HLA-B40:107', 'HLA-B40:108', 'HLA-B40:109', 'HLA-B40:11',
#                             'HLA-B40:110', 'HLA-B40:111', 'HLA-B40:112', 'HLA-B40:113', 'HLA-B40:114', 'HLA-B40:115',
#                             'HLA-B40:116', 'HLA-B40:117', 'HLA-B40:119', 'HLA-B40:12', 'HLA-B40:120', 'HLA-B40:121',
#                             'HLA-B40:122', 'HLA-B40:123', 'HLA-B40:124', 'HLA-B40:125', 'HLA-B40:126', 'HLA-B40:127',
#                             'HLA-B40:128', 'HLA-B40:129', 'HLA-B40:13', 'HLA-B40:130', 'HLA-B40:131', 'HLA-B40:132',
#                             'HLA-B40:134', 'HLA-B40:135', 'HLA-B40:136', 'HLA-B40:137', 'HLA-B40:138', 'HLA-B40:139',
#                             'HLA-B40:14', 'HLA-B40:140', 'HLA-B40:141', 'HLA-B40:143', 'HLA-B40:145', 'HLA-B40:146',
#                             'HLA-B40:147', 'HLA-B40:15', 'HLA-B40:16', 'HLA-B40:18', 'HLA-B40:19', 'HLA-B40:20',
#                             'HLA-B40:21', 'HLA-B40:23', 'HLA-B40:24', 'HLA-B40:25', 'HLA-B40:26', 'HLA-B40:27',
#                             'HLA-B40:28', 'HLA-B40:29', 'HLA-B40:30', 'HLA-B40:31', 'HLA-B40:32', 'HLA-B40:33',
#                             'HLA-B40:34', 'HLA-B40:35', 'HLA-B40:36', 'HLA-B40:37', 'HLA-B40:38', 'HLA-B40:39',
#                             'HLA-B40:40', 'HLA-B40:42', 'HLA-B40:43', 'HLA-B40:44', 'HLA-B40:45', 'HLA-B40:46',
#                             'HLA-B40:47', 'HLA-B40:48', 'HLA-B40:49', 'HLA-B40:50', 'HLA-B40:51', 'HLA-B40:52',
#                             'HLA-B40:53', 'HLA-B40:54', 'HLA-B40:55', 'HLA-B40:56', 'HLA-B40:57', 'HLA-B40:58',
#                             'HLA-B40:59', 'HLA-B40:60', 'HLA-B40:61', 'HLA-B40:62', 'HLA-B40:63', 'HLA-B40:64',
#                             'HLA-B40:65', 'HLA-B40:66', 'HLA-B40:67', 'HLA-B40:68', 'HLA-B40:69', 'HLA-B40:70',
#                             'HLA-B40:71', 'HLA-B40:72', 'HLA-B40:73', 'HLA-B40:74', 'HLA-B40:75', 'HLA-B40:76',
#                             'HLA-B40:77', 'HLA-B40:78', 'HLA-B40:79', 'HLA-B40:80', 'HLA-B40:81', 'HLA-B40:82',
#                             'HLA-B40:83', 'HLA-B40:84', 'HLA-B40:85', 'HLA-B40:86', 'HLA-B40:87', 'HLA-B40:88',
#                             'HLA-B40:89', 'HLA-B40:90', 'HLA-B40:91', 'HLA-B40:92', 'HLA-B40:93', 'HLA-B40:94',
#                             'HLA-B40:95', 'HLA-B40:96', 'HLA-B40:97', 'HLA-B40:98', 'HLA-B40:99', 'HLA-B41:01',
#                             'HLA-B41:02', 'HLA-B41:03', 'HLA-B41:04', 'HLA-B41:05', 'HLA-B41:06', 'HLA-B41:07',
#                             'HLA-B41:08', 'HLA-B41:09', 'HLA-B41:10', 'HLA-B41:11', 'HLA-B41:12', 'HLA-B42:01',
#                             'HLA-B42:02', 'HLA-B42:04', 'HLA-B42:05', 'HLA-B42:06', 'HLA-B42:07', 'HLA-B42:08',
#                             'HLA-B42:09', 'HLA-B42:10', 'HLA-B42:11', 'HLA-B42:12', 'HLA-B42:13', 'HLA-B42:14',
#                             'HLA-B44:02', 'HLA-B44:03', 'HLA-B44:04', 'HLA-B44:05', 'HLA-B44:06', 'HLA-B44:07',
#                             'HLA-B44:08', 'HLA-B44:09', 'HLA-B44:10', 'HLA-B44:100', 'HLA-B44:101', 'HLA-B44:102',
#                             'HLA-B44:103', 'HLA-B44:104', 'HLA-B44:105', 'HLA-B44:106', 'HLA-B44:107', 'HLA-B44:109',
#                             'HLA-B44:11', 'HLA-B44:110', 'HLA-B44:12', 'HLA-B44:13', 'HLA-B44:14', 'HLA-B44:15',
#                             'HLA-B44:16', 'HLA-B44:17', 'HLA-B44:18', 'HLA-B44:20', 'HLA-B44:21', 'HLA-B44:22',
#                             'HLA-B44:24', 'HLA-B44:25', 'HLA-B44:26', 'HLA-B44:27', 'HLA-B44:28', 'HLA-B44:29',
#                             'HLA-B44:30', 'HLA-B44:31', 'HLA-B44:32', 'HLA-B44:33', 'HLA-B44:34', 'HLA-B44:35',
#                             'HLA-B44:36', 'HLA-B44:37', 'HLA-B44:38', 'HLA-B44:39', 'HLA-B44:40', 'HLA-B44:41',
#                             'HLA-B44:42', 'HLA-B44:43', 'HLA-B44:44', 'HLA-B44:45', 'HLA-B44:46', 'HLA-B44:47',
#                             'HLA-B44:48', 'HLA-B44:49', 'HLA-B44:50', 'HLA-B44:51', 'HLA-B44:53', 'HLA-B44:54',
#                             'HLA-B44:55', 'HLA-B44:57', 'HLA-B44:59', 'HLA-B44:60', 'HLA-B44:62', 'HLA-B44:63',
#                             'HLA-B44:64', 'HLA-B44:65', 'HLA-B44:66', 'HLA-B44:67', 'HLA-B44:68', 'HLA-B44:69',
#                             'HLA-B44:70', 'HLA-B44:71', 'HLA-B44:72', 'HLA-B44:73', 'HLA-B44:74', 'HLA-B44:75',
#                             'HLA-B44:76', 'HLA-B44:77', 'HLA-B44:78', 'HLA-B44:79', 'HLA-B44:80', 'HLA-B44:81',
#                             'HLA-B44:82', 'HLA-B44:83', 'HLA-B44:84', 'HLA-B44:85', 'HLA-B44:86', 'HLA-B44:87',
#                             'HLA-B44:88', 'HLA-B44:89', 'HLA-B44:90', 'HLA-B44:91', 'HLA-B44:92', 'HLA-B44:93',
#                             'HLA-B44:94', 'HLA-B44:95', 'HLA-B44:96', 'HLA-B44:97', 'HLA-B44:98', 'HLA-B44:99',
#                             'HLA-B45:01', 'HLA-B45:02', 'HLA-B45:03', 'HLA-B45:04', 'HLA-B45:05', 'HLA-B45:06',
#                             'HLA-B45:07', 'HLA-B45:08', 'HLA-B45:09', 'HLA-B45:10', 'HLA-B45:11', 'HLA-B45:12',
#                             'HLA-B46:01', 'HLA-B46:02', 'HLA-B46:03', 'HLA-B46:04', 'HLA-B46:05', 'HLA-B46:06',
#                             'HLA-B46:08', 'HLA-B46:09', 'HLA-B46:10', 'HLA-B46:11', 'HLA-B46:12', 'HLA-B46:13',
#                             'HLA-B46:14', 'HLA-B46:16', 'HLA-B46:17', 'HLA-B46:18', 'HLA-B46:19', 'HLA-B46:20',
#                             'HLA-B46:21', 'HLA-B46:22', 'HLA-B46:23', 'HLA-B46:24', 'HLA-B47:01', 'HLA-B47:02',
#                             'HLA-B47:03', 'HLA-B47:04', 'HLA-B47:05', 'HLA-B47:06', 'HLA-B47:07', 'HLA-B48:01',
#                             'HLA-B48:02', 'HLA-B48:03', 'HLA-B48:04', 'HLA-B48:05', 'HLA-B48:06', 'HLA-B48:07',
#                             'HLA-B48:08', 'HLA-B48:09', 'HLA-B48:10', 'HLA-B48:11', 'HLA-B48:12', 'HLA-B48:13',
#                             'HLA-B48:14', 'HLA-B48:15', 'HLA-B48:16', 'HLA-B48:17', 'HLA-B48:18', 'HLA-B48:19',
#                             'HLA-B48:20', 'HLA-B48:21', 'HLA-B48:22', 'HLA-B48:23', 'HLA-B49:01', 'HLA-B49:02',
#                             'HLA-B49:03', 'HLA-B49:04', 'HLA-B49:05', 'HLA-B49:06', 'HLA-B49:07', 'HLA-B49:08',
#                             'HLA-B49:09', 'HLA-B49:10', 'HLA-B50:01', 'HLA-B50:02', 'HLA-B50:04', 'HLA-B50:05',
#                             'HLA-B50:06', 'HLA-B50:07', 'HLA-B50:08', 'HLA-B50:09', 'HLA-B51:01', 'HLA-B51:02',
#                             'HLA-B51:03', 'HLA-B51:04', 'HLA-B51:05', 'HLA-B51:06', 'HLA-B51:07', 'HLA-B51:08',
#                             'HLA-B51:09', 'HLA-B51:12', 'HLA-B51:13', 'HLA-B51:14', 'HLA-B51:15', 'HLA-B51:16',
#                             'HLA-B51:17', 'HLA-B51:18', 'HLA-B51:19', 'HLA-B51:20', 'HLA-B51:21', 'HLA-B51:22',
#                             'HLA-B51:23', 'HLA-B51:24', 'HLA-B51:26', 'HLA-B51:28', 'HLA-B51:29', 'HLA-B51:30',
#                             'HLA-B51:31', 'HLA-B51:32', 'HLA-B51:33', 'HLA-B51:34', 'HLA-B51:35', 'HLA-B51:36',
#                             'HLA-B51:37', 'HLA-B51:38', 'HLA-B51:39', 'HLA-B51:40', 'HLA-B51:42', 'HLA-B51:43',
#                             'HLA-B51:45', 'HLA-B51:46', 'HLA-B51:48', 'HLA-B51:49', 'HLA-B51:50', 'HLA-B51:51',
#                             'HLA-B51:52', 'HLA-B51:53', 'HLA-B51:54', 'HLA-B51:55', 'HLA-B51:56', 'HLA-B51:57',
#                             'HLA-B51:58', 'HLA-B51:59', 'HLA-B51:60', 'HLA-B51:61', 'HLA-B51:62', 'HLA-B51:63',
#                             'HLA-B51:64', 'HLA-B51:65', 'HLA-B51:66', 'HLA-B51:67', 'HLA-B51:68', 'HLA-B51:69',
#                             'HLA-B51:70', 'HLA-B51:71', 'HLA-B51:72', 'HLA-B51:73', 'HLA-B51:74', 'HLA-B51:75',
#                             'HLA-B51:76', 'HLA-B51:77', 'HLA-B51:78', 'HLA-B51:79', 'HLA-B51:80', 'HLA-B51:81',
#                             'HLA-B51:82', 'HLA-B51:83', 'HLA-B51:84', 'HLA-B51:85', 'HLA-B51:86', 'HLA-B51:87',
#                             'HLA-B51:88', 'HLA-B51:89', 'HLA-B51:90', 'HLA-B51:91', 'HLA-B51:92', 'HLA-B51:93',
#                             'HLA-B51:94', 'HLA-B51:95', 'HLA-B51:96', 'HLA-B52:01', 'HLA-B52:02', 'HLA-B52:03',
#                             'HLA-B52:04', 'HLA-B52:05', 'HLA-B52:06', 'HLA-B52:07', 'HLA-B52:08', 'HLA-B52:09',
#                             'HLA-B52:10', 'HLA-B52:11', 'HLA-B52:12', 'HLA-B52:13', 'HLA-B52:14', 'HLA-B52:15',
#                             'HLA-B52:16', 'HLA-B52:17', 'HLA-B52:18', 'HLA-B52:19', 'HLA-B52:20', 'HLA-B52:21',
#                             'HLA-B53:01', 'HLA-B53:02', 'HLA-B53:03', 'HLA-B53:04', 'HLA-B53:05', 'HLA-B53:06',
#                             'HLA-B53:07', 'HLA-B53:08', 'HLA-B53:09', 'HLA-B53:10', 'HLA-B53:11', 'HLA-B53:12',
#                             'HLA-B53:13', 'HLA-B53:14', 'HLA-B53:15', 'HLA-B53:16', 'HLA-B53:17', 'HLA-B53:18',
#                             'HLA-B53:19', 'HLA-B53:20', 'HLA-B53:21', 'HLA-B53:22', 'HLA-B53:23', 'HLA-B54:01',
#                             'HLA-B54:02', 'HLA-B54:03', 'HLA-B54:04', 'HLA-B54:06', 'HLA-B54:07', 'HLA-B54:09',
#                             'HLA-B54:10', 'HLA-B54:11', 'HLA-B54:12', 'HLA-B54:13', 'HLA-B54:14', 'HLA-B54:15',
#                             'HLA-B54:16', 'HLA-B54:17', 'HLA-B54:18', 'HLA-B54:19', 'HLA-B54:20', 'HLA-B54:21',
#                             'HLA-B54:22', 'HLA-B54:23', 'HLA-B55:01', 'HLA-B55:02', 'HLA-B55:03', 'HLA-B55:04',
#                             'HLA-B55:05', 'HLA-B55:07', 'HLA-B55:08', 'HLA-B55:09', 'HLA-B55:10', 'HLA-B55:11',
#                             'HLA-B55:12', 'HLA-B55:13', 'HLA-B55:14', 'HLA-B55:15', 'HLA-B55:16', 'HLA-B55:17',
#                             'HLA-B55:18', 'HLA-B55:19', 'HLA-B55:20', 'HLA-B55:21', 'HLA-B55:22', 'HLA-B55:23',
#                             'HLA-B55:24', 'HLA-B55:25', 'HLA-B55:26', 'HLA-B55:27', 'HLA-B55:28', 'HLA-B55:29',
#                             'HLA-B55:30', 'HLA-B55:31', 'HLA-B55:32', 'HLA-B55:33', 'HLA-B55:34', 'HLA-B55:35',
#                             'HLA-B55:36', 'HLA-B55:37', 'HLA-B55:38', 'HLA-B55:39', 'HLA-B55:40', 'HLA-B55:41',
#                             'HLA-B55:42', 'HLA-B55:43', 'HLA-B56:01', 'HLA-B56:02', 'HLA-B56:03', 'HLA-B56:04',
#                             'HLA-B56:05', 'HLA-B56:06', 'HLA-B56:07', 'HLA-B56:08', 'HLA-B56:09', 'HLA-B56:10',
#                             'HLA-B56:11', 'HLA-B56:12', 'HLA-B56:13', 'HLA-B56:14', 'HLA-B56:15', 'HLA-B56:16',
#                             'HLA-B56:17', 'HLA-B56:18', 'HLA-B56:20', 'HLA-B56:21', 'HLA-B56:22', 'HLA-B56:23',
#                             'HLA-B56:24', 'HLA-B56:25', 'HLA-B56:26', 'HLA-B56:27', 'HLA-B56:29', 'HLA-B57:01',
#                             'HLA-B57:02', 'HLA-B57:03', 'HLA-B57:04', 'HLA-B57:05', 'HLA-B57:06', 'HLA-B57:07',
#                             'HLA-B57:08', 'HLA-B57:09', 'HLA-B57:10', 'HLA-B57:11', 'HLA-B57:12', 'HLA-B57:13',
#                             'HLA-B57:14', 'HLA-B57:15', 'HLA-B57:16', 'HLA-B57:17', 'HLA-B57:18', 'HLA-B57:19',
#                             'HLA-B57:20', 'HLA-B57:21', 'HLA-B57:22', 'HLA-B57:23', 'HLA-B57:24', 'HLA-B57:25',
#                             'HLA-B57:26', 'HLA-B57:27', 'HLA-B57:29', 'HLA-B57:30', 'HLA-B57:31', 'HLA-B57:32',
#                             'HLA-B58:01', 'HLA-B58:02', 'HLA-B58:04', 'HLA-B58:05', 'HLA-B58:06', 'HLA-B58:07',
#                             'HLA-B58:08', 'HLA-B58:09', 'HLA-B58:11', 'HLA-B58:12', 'HLA-B58:13', 'HLA-B58:14',
#                             'HLA-B58:15', 'HLA-B58:16', 'HLA-B58:18', 'HLA-B58:19', 'HLA-B58:20', 'HLA-B58:21',
#                             'HLA-B58:22', 'HLA-B58:23', 'HLA-B58:24', 'HLA-B58:25', 'HLA-B58:26', 'HLA-B58:27',
#                             'HLA-B58:28', 'HLA-B58:29', 'HLA-B58:30', 'HLA-B59:01', 'HLA-B59:02', 'HLA-B59:03',
#                             'HLA-B59:04', 'HLA-B59:05', 'HLA-B67:01', 'HLA-B67:02', 'HLA-B73:01', 'HLA-B73:02',
#                             'HLA-B78:01', 'HLA-B78:02', 'HLA-B78:03', 'HLA-B78:04', 'HLA-B78:05', 'HLA-B78:06',
#                             'HLA-B78:07', 'HLA-B81:01', 'HLA-B81:02', 'HLA-B81:03', 'HLA-B81:05', 'HLA-B82:01',
#                             'HLA-B82:02', 'HLA-B82:03', 'HLA-B83:01', 'HLA-C01:02', 'HLA-C01:03', 'HLA-C01:04',
#                             'HLA-C01:05', 'HLA-C01:06', 'HLA-C01:07', 'HLA-C01:08', 'HLA-C01:09', 'HLA-C01:10',
#                             'HLA-C01:11', 'HLA-C01:12', 'HLA-C01:13', 'HLA-C01:14', 'HLA-C01:15', 'HLA-C01:16',
#                             'HLA-C01:17', 'HLA-C01:18', 'HLA-C01:19', 'HLA-C01:20', 'HLA-C01:21', 'HLA-C01:22',
#                             'HLA-C01:23', 'HLA-C01:24', 'HLA-C01:25', 'HLA-C01:26', 'HLA-C01:27', 'HLA-C01:28',
#                             'HLA-C01:29', 'HLA-C01:30', 'HLA-C01:31', 'HLA-C01:32', 'HLA-C01:33', 'HLA-C01:34',
#                             'HLA-C01:35', 'HLA-C01:36', 'HLA-C01:38', 'HLA-C01:39', 'HLA-C01:40', 'HLA-C02:02',
#                             'HLA-C02:03', 'HLA-C02:04', 'HLA-C02:05', 'HLA-C02:06', 'HLA-C02:07', 'HLA-C02:08',
#                             'HLA-C02:09', 'HLA-C02:10', 'HLA-C02:11', 'HLA-C02:12', 'HLA-C02:13', 'HLA-C02:14',
#                             'HLA-C02:15', 'HLA-C02:16', 'HLA-C02:17', 'HLA-C02:18', 'HLA-C02:19', 'HLA-C02:20',
#                             'HLA-C02:21', 'HLA-C02:22', 'HLA-C02:23', 'HLA-C02:24', 'HLA-C02:26', 'HLA-C02:27',
#                             'HLA-C02:28', 'HLA-C02:29', 'HLA-C02:30', 'HLA-C02:31', 'HLA-C02:32', 'HLA-C02:33',
#                             'HLA-C02:34', 'HLA-C02:35', 'HLA-C02:36', 'HLA-C02:37', 'HLA-C02:39', 'HLA-C02:40',
#                             'HLA-C03:01', 'HLA-C03:02', 'HLA-C03:03', 'HLA-C03:04', 'HLA-C03:05', 'HLA-C03:06',
#                             'HLA-C03:07', 'HLA-C03:08', 'HLA-C03:09', 'HLA-C03:10', 'HLA-C03:11', 'HLA-C03:12',
#                             'HLA-C03:13', 'HLA-C03:14', 'HLA-C03:15', 'HLA-C03:16', 'HLA-C03:17', 'HLA-C03:18',
#                             'HLA-C03:19', 'HLA-C03:21', 'HLA-C03:23', 'HLA-C03:24', 'HLA-C03:25', 'HLA-C03:26',
#                             'HLA-C03:27', 'HLA-C03:28', 'HLA-C03:29', 'HLA-C03:30', 'HLA-C03:31', 'HLA-C03:32',
#                             'HLA-C03:33', 'HLA-C03:34', 'HLA-C03:35', 'HLA-C03:36', 'HLA-C03:37', 'HLA-C03:38',
#                             'HLA-C03:39', 'HLA-C03:40', 'HLA-C03:41', 'HLA-C03:42', 'HLA-C03:43', 'HLA-C03:44',
#                             'HLA-C03:45', 'HLA-C03:46', 'HLA-C03:47', 'HLA-C03:48', 'HLA-C03:49', 'HLA-C03:50',
#                             'HLA-C03:51', 'HLA-C03:52', 'HLA-C03:53', 'HLA-C03:54', 'HLA-C03:55', 'HLA-C03:56',
#                             'HLA-C03:57', 'HLA-C03:58', 'HLA-C03:59', 'HLA-C03:60', 'HLA-C03:61', 'HLA-C03:62',
#                             'HLA-C03:63', 'HLA-C03:64', 'HLA-C03:65', 'HLA-C03:66', 'HLA-C03:67', 'HLA-C03:68',
#                             'HLA-C03:69', 'HLA-C03:70', 'HLA-C03:71', 'HLA-C03:72', 'HLA-C03:73', 'HLA-C03:74',
#                             'HLA-C03:75', 'HLA-C03:76', 'HLA-C03:77', 'HLA-C03:78', 'HLA-C03:79', 'HLA-C03:80',
#                             'HLA-C03:81', 'HLA-C03:82', 'HLA-C03:83', 'HLA-C03:84', 'HLA-C03:85', 'HLA-C03:86',
#                             'HLA-C03:87', 'HLA-C03:88', 'HLA-C03:89', 'HLA-C03:90', 'HLA-C03:91', 'HLA-C03:92',
#                             'HLA-C03:93', 'HLA-C03:94', 'HLA-C04:01', 'HLA-C04:03', 'HLA-C04:04', 'HLA-C04:05',
#                             'HLA-C04:06', 'HLA-C04:07', 'HLA-C04:08', 'HLA-C04:10', 'HLA-C04:11', 'HLA-C04:12',
#                             'HLA-C04:13', 'HLA-C04:14', 'HLA-C04:15', 'HLA-C04:16', 'HLA-C04:17', 'HLA-C04:18',
#                             'HLA-C04:19', 'HLA-C04:20', 'HLA-C04:23', 'HLA-C04:24', 'HLA-C04:25', 'HLA-C04:26',
#                             'HLA-C04:27', 'HLA-C04:28', 'HLA-C04:29', 'HLA-C04:30', 'HLA-C04:31', 'HLA-C04:32',
#                             'HLA-C04:33', 'HLA-C04:34', 'HLA-C04:35', 'HLA-C04:36', 'HLA-C04:37', 'HLA-C04:38',
#                             'HLA-C04:39', 'HLA-C04:40', 'HLA-C04:41', 'HLA-C04:42', 'HLA-C04:43', 'HLA-C04:44',
#                             'HLA-C04:45', 'HLA-C04:46', 'HLA-C04:47', 'HLA-C04:48', 'HLA-C04:49', 'HLA-C04:50',
#                             'HLA-C04:51', 'HLA-C04:52', 'HLA-C04:53', 'HLA-C04:54', 'HLA-C04:55', 'HLA-C04:56',
#                             'HLA-C04:57', 'HLA-C04:58', 'HLA-C04:60', 'HLA-C04:61', 'HLA-C04:62', 'HLA-C04:63',
#                             'HLA-C04:64', 'HLA-C04:65', 'HLA-C04:66', 'HLA-C04:67', 'HLA-C04:68', 'HLA-C04:69',
#                             'HLA-C04:70', 'HLA-C05:01', 'HLA-C05:03', 'HLA-C05:04', 'HLA-C05:05', 'HLA-C05:06',
#                             'HLA-C05:08', 'HLA-C05:09', 'HLA-C05:10', 'HLA-C05:11', 'HLA-C05:12', 'HLA-C05:13',
#                             'HLA-C05:14', 'HLA-C05:15', 'HLA-C05:16', 'HLA-C05:17', 'HLA-C05:18', 'HLA-C05:19',
#                             'HLA-C05:20', 'HLA-C05:21', 'HLA-C05:22', 'HLA-C05:23', 'HLA-C05:24', 'HLA-C05:25',
#                             'HLA-C05:26', 'HLA-C05:27', 'HLA-C05:28', 'HLA-C05:29', 'HLA-C05:30', 'HLA-C05:31',
#                             'HLA-C05:32', 'HLA-C05:33', 'HLA-C05:34', 'HLA-C05:35', 'HLA-C05:36', 'HLA-C05:37',
#                             'HLA-C05:38', 'HLA-C05:39', 'HLA-C05:40', 'HLA-C05:41', 'HLA-C05:42', 'HLA-C05:43',
#                             'HLA-C05:44', 'HLA-C05:45', 'HLA-C06:02', 'HLA-C06:03', 'HLA-C06:04', 'HLA-C06:05',
#                             'HLA-C06:06', 'HLA-C06:07', 'HLA-C06:08', 'HLA-C06:09', 'HLA-C06:10', 'HLA-C06:11',
#                             'HLA-C06:12', 'HLA-C06:13', 'HLA-C06:14', 'HLA-C06:15', 'HLA-C06:17', 'HLA-C06:18',
#                             'HLA-C06:19', 'HLA-C06:20', 'HLA-C06:21', 'HLA-C06:22', 'HLA-C06:23', 'HLA-C06:24',
#                             'HLA-C06:25', 'HLA-C06:26', 'HLA-C06:27', 'HLA-C06:28', 'HLA-C06:29', 'HLA-C06:30',
#                             'HLA-C06:31', 'HLA-C06:32', 'HLA-C06:33', 'HLA-C06:34', 'HLA-C06:35', 'HLA-C06:36',
#                             'HLA-C06:37', 'HLA-C06:38', 'HLA-C06:39', 'HLA-C06:40', 'HLA-C06:41', 'HLA-C06:42',
#                             'HLA-C06:43', 'HLA-C06:44', 'HLA-C06:45', 'HLA-C07:01', 'HLA-C07:02', 'HLA-C07:03',
#                             'HLA-C07:04', 'HLA-C07:05', 'HLA-C07:06', 'HLA-C07:07', 'HLA-C07:08', 'HLA-C07:09',
#                             'HLA-C07:10', 'HLA-C07:100', 'HLA-C07:101', 'HLA-C07:102', 'HLA-C07:103', 'HLA-C07:105',
#                             'HLA-C07:106', 'HLA-C07:107', 'HLA-C07:108', 'HLA-C07:109', 'HLA-C07:11', 'HLA-C07:110',
#                             'HLA-C07:111', 'HLA-C07:112', 'HLA-C07:113', 'HLA-C07:114', 'HLA-C07:115', 'HLA-C07:116',
#                             'HLA-C07:117', 'HLA-C07:118', 'HLA-C07:119', 'HLA-C07:12', 'HLA-C07:120', 'HLA-C07:122',
#                             'HLA-C07:123', 'HLA-C07:124', 'HLA-C07:125', 'HLA-C07:126', 'HLA-C07:127', 'HLA-C07:128',
#                             'HLA-C07:129', 'HLA-C07:13', 'HLA-C07:130', 'HLA-C07:131', 'HLA-C07:132', 'HLA-C07:133',
#                             'HLA-C07:134', 'HLA-C07:135', 'HLA-C07:136', 'HLA-C07:137', 'HLA-C07:138', 'HLA-C07:139',
#                             'HLA-C07:14', 'HLA-C07:140', 'HLA-C07:141', 'HLA-C07:142', 'HLA-C07:143', 'HLA-C07:144',
#                             'HLA-C07:145', 'HLA-C07:146', 'HLA-C07:147', 'HLA-C07:148', 'HLA-C07:149', 'HLA-C07:15',
#                             'HLA-C07:16', 'HLA-C07:17', 'HLA-C07:18', 'HLA-C07:19', 'HLA-C07:20', 'HLA-C07:21',
#                             'HLA-C07:22', 'HLA-C07:23', 'HLA-C07:24', 'HLA-C07:25', 'HLA-C07:26', 'HLA-C07:27',
#                             'HLA-C07:28', 'HLA-C07:29', 'HLA-C07:30', 'HLA-C07:31', 'HLA-C07:35', 'HLA-C07:36',
#                             'HLA-C07:37', 'HLA-C07:38', 'HLA-C07:39', 'HLA-C07:40', 'HLA-C07:41', 'HLA-C07:42',
#                             'HLA-C07:43', 'HLA-C07:44', 'HLA-C07:45', 'HLA-C07:46', 'HLA-C07:47', 'HLA-C07:48',
#                             'HLA-C07:49', 'HLA-C07:50', 'HLA-C07:51', 'HLA-C07:52', 'HLA-C07:53', 'HLA-C07:54',
#                             'HLA-C07:56', 'HLA-C07:57', 'HLA-C07:58', 'HLA-C07:59', 'HLA-C07:60', 'HLA-C07:62',
#                             'HLA-C07:63', 'HLA-C07:64', 'HLA-C07:65', 'HLA-C07:66', 'HLA-C07:67', 'HLA-C07:68',
#                             'HLA-C07:69', 'HLA-C07:70', 'HLA-C07:71', 'HLA-C07:72', 'HLA-C07:73', 'HLA-C07:74',
#                             'HLA-C07:75', 'HLA-C07:76', 'HLA-C07:77', 'HLA-C07:78', 'HLA-C07:79', 'HLA-C07:80',
#                             'HLA-C07:81', 'HLA-C07:82', 'HLA-C07:83', 'HLA-C07:84', 'HLA-C07:85', 'HLA-C07:86',
#                             'HLA-C07:87', 'HLA-C07:88', 'HLA-C07:89', 'HLA-C07:90', 'HLA-C07:91', 'HLA-C07:92',
#                             'HLA-C07:93', 'HLA-C07:94', 'HLA-C07:95', 'HLA-C07:96', 'HLA-C07:97', 'HLA-C07:99',
#                             'HLA-C08:01', 'HLA-C08:02', 'HLA-C08:03', 'HLA-C08:04', 'HLA-C08:05', 'HLA-C08:06',
#                             'HLA-C08:07', 'HLA-C08:08', 'HLA-C08:09', 'HLA-C08:10', 'HLA-C08:11', 'HLA-C08:12',
#                             'HLA-C08:13', 'HLA-C08:14', 'HLA-C08:15', 'HLA-C08:16', 'HLA-C08:17', 'HLA-C08:18',
#                             'HLA-C08:19', 'HLA-C08:20', 'HLA-C08:21', 'HLA-C08:22', 'HLA-C08:23', 'HLA-C08:24',
#                             'HLA-C08:25', 'HLA-C08:27', 'HLA-C08:28', 'HLA-C08:29', 'HLA-C08:30', 'HLA-C08:31',
#                             'HLA-C08:32', 'HLA-C08:33', 'HLA-C08:34', 'HLA-C08:35', 'HLA-C12:02', 'HLA-C12:03',
#                             'HLA-C12:04', 'HLA-C12:05', 'HLA-C12:06', 'HLA-C12:07', 'HLA-C12:08', 'HLA-C12:09',
#                             'HLA-C12:10', 'HLA-C12:11', 'HLA-C12:12', 'HLA-C12:13', 'HLA-C12:14', 'HLA-C12:15',
#                             'HLA-C12:16', 'HLA-C12:17', 'HLA-C12:18', 'HLA-C12:19', 'HLA-C12:20', 'HLA-C12:21',
#                             'HLA-C12:22', 'HLA-C12:23', 'HLA-C12:24', 'HLA-C12:25', 'HLA-C12:26', 'HLA-C12:27',
#                             'HLA-C12:28', 'HLA-C12:29', 'HLA-C12:30', 'HLA-C12:31', 'HLA-C12:32', 'HLA-C12:33',
#                             'HLA-C12:34', 'HLA-C12:35', 'HLA-C12:36', 'HLA-C12:37', 'HLA-C12:38', 'HLA-C12:40',
#                             'HLA-C12:41', 'HLA-C12:43', 'HLA-C12:44', 'HLA-C14:02', 'HLA-C14:03', 'HLA-C14:04',
#                             'HLA-C14:05', 'HLA-C14:06', 'HLA-C14:08', 'HLA-C14:09', 'HLA-C14:10', 'HLA-C14:11',
#                             'HLA-C14:12', 'HLA-C14:13', 'HLA-C14:14', 'HLA-C14:15', 'HLA-C14:16', 'HLA-C14:17',
#                             'HLA-C14:18', 'HLA-C14:19', 'HLA-C14:20', 'HLA-C15:02', 'HLA-C15:03', 'HLA-C15:04',
#                             'HLA-C15:05', 'HLA-C15:06', 'HLA-C15:07', 'HLA-C15:08', 'HLA-C15:09', 'HLA-C15:10',
#                             'HLA-C15:11', 'HLA-C15:12', 'HLA-C15:13', 'HLA-C15:15', 'HLA-C15:16', 'HLA-C15:17',
#                             'HLA-C15:18', 'HLA-C15:19', 'HLA-C15:20', 'HLA-C15:21', 'HLA-C15:22', 'HLA-C15:23',
#                             'HLA-C15:24', 'HLA-C15:25', 'HLA-C15:26', 'HLA-C15:27', 'HLA-C15:28', 'HLA-C15:29',
#                             'HLA-C15:30', 'HLA-C15:31', 'HLA-C15:33', 'HLA-C15:34', 'HLA-C15:35', 'HLA-C16:01',
#                             'HLA-C16:02', 'HLA-C16:04', 'HLA-C16:06', 'HLA-C16:07', 'HLA-C16:08', 'HLA-C16:09',
#                             'HLA-C16:10', 'HLA-C16:11', 'HLA-C16:12', 'HLA-C16:13', 'HLA-C16:14', 'HLA-C16:15',
#                             'HLA-C16:17', 'HLA-C16:18', 'HLA-C16:19', 'HLA-C16:20', 'HLA-C16:21', 'HLA-C16:22',
#                             'HLA-C16:23', 'HLA-C16:24', 'HLA-C16:25', 'HLA-C16:26', 'HLA-C17:01', 'HLA-C17:02',
#                             'HLA-C17:03', 'HLA-C17:04', 'HLA-C17:05', 'HLA-C17:06', 'HLA-C17:07', 'HLA-C18:01',
#                             'HLA-C18:02', 'HLA-C18:03', 'HLA-E01:01', 'HLA-G01:01', 'HLA-G01:02', 'HLA-G01:03',
#                             'HLA-G01:04', 'HLA-G01:06', 'HLA-G01:07', 'HLA-G01:08', 'HLA-G01:09', 'Mamu-A1:00101',
#                             'Mamu-A1:00102', 'Mamu-A1:00103', 'Mamu-A1:00104', 'Mamu-A1:00105', 'Mamu-A1:00201', 'Mamu-A1:00301',
#                             'Mamu-A1:00302', 'Mamu-A1:00303', 'Mamu-A1:00304', 'Mamu-A1:00305', 'Mamu-A1:00306', 'Mamu-A1:00307',
#                             'Mamu-A1:00308', 'Mamu-A1:00310', 'Mamu-A1:00401', 'Mamu-A1:00402', 'Mamu-A1:00403', 'Mamu-A1:00601',
#                             'Mamu-A1:00602', 'Mamu-A1:00701', 'Mamu-A1:00702', 'Mamu-A1:00703', 'Mamu-A1:00704', 'Mamu-A1:00705',
#                             'Mamu-A1:00801', 'Mamu-A1:01001', 'Mamu-A1:01002', 'Mamu-A1:01101', 'Mamu-A1:01102', 'Mamu-A1:01103',
#                             'Mamu-A1:01104', 'Mamu-A1:01201', 'Mamu-A1:01601', 'Mamu-A1:01801', 'Mamu-A1:01802', 'Mamu-A1:01803',
#                             'Mamu-A1:01804', 'Mamu-A1:01805', 'Mamu-A1:01806', 'Mamu-A1:01807', 'Mamu-A1:01808', 'Mamu-A1:01901',
#                             'Mamu-A1:01902', 'Mamu-A1:01903', 'Mamu-A1:01904', 'Mamu-A1:01905', 'Mamu-A1:01906', 'Mamu-A1:01907',
#                             'Mamu-A1:02201', 'Mamu-A1:02202', 'Mamu-A1:02203', 'Mamu-A1:02301', 'Mamu-A1:02302', 'Mamu-A1:02501',
#                             'Mamu-A1:02502', 'Mamu-A1:02601', 'Mamu-A1:02602', 'Mamu-A1:02603', 'Mamu-A1:02801', 'Mamu-A1:02802',
#                             'Mamu-A1:02803', 'Mamu-A1:02804', 'Mamu-A1:02805', 'Mamu-A1:02806', 'Mamu-A1:03201', 'Mamu-A1:03202',
#                             'Mamu-A1:03203', 'Mamu-A1:03301', 'Mamu-A1:04001', 'Mamu-A1:04002', 'Mamu-A1:04003', 'Mamu-A1:04101',
#                             'Mamu-A1:04102', 'Mamu-A1:04201', 'Mamu-A1:04301', 'Mamu-A1:04501', 'Mamu-A1:04801', 'Mamu-A1:04901',
#                             'Mamu-A1:04902', 'Mamu-A1:04903', 'Mamu-A1:04904', 'Mamu-A1:05001', 'Mamu-A1:05101', 'Mamu-A1:05201',
#                             'Mamu-A1:05301', 'Mamu-A1:05302', 'Mamu-A1:05401', 'Mamu-A1:05402', 'Mamu-A1:05501', 'Mamu-A1:05601',
#                             'Mamu-A1:05602', 'Mamu-A1:05603', 'Mamu-A1:05701', 'Mamu-A1:05702', 'Mamu-A1:05901', 'Mamu-A1:06001',
#                             'Mamu-A1:06101', 'Mamu-A1:06301', 'Mamu-A1:06501', 'Mamu-A1:06601', 'Mamu-A1:07301', 'Mamu-A1:07401',
#                             'Mamu-A1:07402', 'Mamu-A1:07403', 'Mamu-A1:08101', 'Mamu-A1:08501', 'Mamu-A1:09101', 'Mamu-A1:09201',
#                             'Mamu-A1:10501', 'Mamu-A1:10502', 'Mamu-A1:10503', 'Mamu-A1:10504', 'Mamu-A1:10601', 'Mamu-A1:10701',
#                             'Mamu-A1:10801', 'Mamu-A1:10901', 'Mamu-A1:11001', 'Mamu-A1:11101', 'Mamu-A1:11201', 'Mamu-A1:11301',
#                             'Mamu-A2:0101', 'Mamu-A2:0102', 'Mamu-A2:0103', 'Mamu-A2:0501', 'Mamu-A2:05020', 'Mamu-A2:05030',
#                             'Mamu-A2:05040', 'Mamu-A2:0505', 'Mamu-A2:0506', 'Mamu-A2:0507', 'Mamu-A2:0509', 'Mamu-A2:0510',
#                             'Mamu-A2:0511', 'Mamu-A2:0512', 'Mamu-A2:0513', 'Mamu-A2:0514', 'Mamu-A2:05150', 'Mamu-A2:05160',
#                             'Mamu-A2:0517', 'Mamu-A2:0518', 'Mamu-A2:0519', 'Mamu-A2:0520', 'Mamu-A2:0521', 'Mamu-A2:0522',
#                             'Mamu-A2:0523', 'Mamu-A2:0524', 'Mamu-A2:0525', 'Mamu-A2:0526', 'Mamu-A2:0527', 'Mamu-A2:0528',
#                             'Mamu-A2:0529', 'Mamu-A2:0531', 'Mamu-A2:05320', 'Mamu-A2:0533', 'Mamu-A2:0534', 'Mamu-A2:0535',
#                             'Mamu-A2:0536', 'Mamu-A2:0537', 'Mamu-A2:0538', 'Mamu-A2:0539', 'Mamu-A2:0540', 'Mamu-A2:0541',
#                             'Mamu-A2:0542', 'Mamu-A2:0543', 'Mamu-A2:0544', 'Mamu-A2:0545', 'Mamu-A2:0546', 'Mamu-A2:0547',
#                             'Mamu-A2:2401', 'Mamu-A2:2402', 'Mamu-A2:2403', 'Mamu-A3:1301', 'Mamu-A3:1302', 'Mamu-A3:1303',
#                             'Mamu-A3:1304', 'Mamu-A3:1305', 'Mamu-A3:1306', 'Mamu-A3:1307', 'Mamu-A3:1308', 'Mamu-A3:1309',
#                             'Mamu-A3:1310', 'Mamu-A3:1311', 'Mamu-A3:1312', 'Mamu-A3:1313', 'Mamu-A4:0101', 'Mamu-A4:01020',
#                             'Mamu-A4:0103', 'Mamu-A4:0202', 'Mamu-A4:0203', 'Mamu-A4:0205', 'Mamu-A4:0301', 'Mamu-A4:0302',
#                             'Mamu-A4:1402', 'Mamu-A4:14030', 'Mamu-A4:1404', 'Mamu-A4:1405', 'Mamu-A4:1406', 'Mamu-A4:1407',
#                             'Mamu-A4:1408', 'Mamu-A4:1409', 'Mamu-A5:30010', 'Mamu-A5:3002', 'Mamu-A5:3003', 'Mamu-A5:3004',
#                             'Mamu-A5:3005', 'Mamu-A5:3006', 'Mamu-A6:0101', 'Mamu-A6:0102', 'Mamu-A6:0103', 'Mamu-A6:0104',
#                             'Mamu-A6:0105', 'Mamu-A7:0101', 'Mamu-A7:0102', 'Mamu-A7:0103', 'Mamu-A7:0201', 'Mamu-AG:01',
#                             'Mamu-AG:02011', 'Mamu-AG:02012', 'Mamu-AG:0202', 'Mamu-AG:03011', 'Mamu-AG:0302', 'Mamu-B:00101',
#                             'Mamu-B:00102', 'Mamu-B:00201', 'Mamu-B:00202', 'Mamu-B:00301', 'Mamu-B:00302', 'Mamu-B:00401',
#                             'Mamu-B:00501', 'Mamu-B:00502', 'Mamu-B:00601', 'Mamu-B:00602', 'Mamu-B:00701', 'Mamu-B:00702',
#                             'Mamu-B:00703', 'Mamu-B:00704', 'Mamu-B:00801', 'Mamu-B:01001', 'Mamu-B:01101', 'Mamu-B:01201',
#                             'Mamu-B:01301', 'Mamu-B:01401', 'Mamu-B:01501', 'Mamu-B:01502', 'Mamu-B:01601', 'Mamu-B:01701',
#                             'Mamu-B:01702', 'Mamu-B:01703', 'Mamu-B:01801', 'Mamu-B:01901', 'Mamu-B:01902', 'Mamu-B:01903',
#                             'Mamu-B:02001', 'Mamu-B:02101', 'Mamu-B:02102', 'Mamu-B:02103', 'Mamu-B:02201', 'Mamu-B:02301',
#                             'Mamu-B:02401', 'Mamu-B:02501', 'Mamu-B:02601', 'Mamu-B:02602', 'Mamu-B:02701', 'Mamu-B:02702',
#                             'Mamu-B:02703', 'Mamu-B:02801', 'Mamu-B:02802', 'Mamu-B:02803', 'Mamu-B:02901', 'Mamu-B:02902',
#                             'Mamu-B:03001', 'Mamu-B:03002', 'Mamu-B:03003', 'Mamu-B:03004', 'Mamu-B:03005', 'Mamu-B:03101',
#                             'Mamu-B:03102', 'Mamu-B:03103', 'Mamu-B:03201', 'Mamu-B:03301', 'Mamu-B:03401', 'Mamu-B:03501',
#                             'Mamu-B:03601', 'Mamu-B:03602', 'Mamu-B:03701', 'Mamu-B:03801', 'Mamu-B:03802', 'Mamu-B:03901',
#                             'Mamu-B:04001', 'Mamu-B:04002', 'Mamu-B:04101', 'Mamu-B:04201', 'Mamu-B:04301', 'Mamu-B:04401',
#                             'Mamu-B:04402', 'Mamu-B:04403', 'Mamu-B:04404', 'Mamu-B:04405', 'Mamu-B:04501', 'Mamu-B:04502',
#                             'Mamu-B:04503', 'Mamu-B:04504', 'Mamu-B:04601', 'Mamu-B:04602', 'Mamu-B:04603', 'Mamu-B:04604',
#                             'Mamu-B:04605', 'Mamu-B:04607', 'Mamu-B:04608', 'Mamu-B:04609', 'Mamu-B:04610', 'Mamu-B:04611',
#                             'Mamu-B:04612', 'Mamu-B:04613', 'Mamu-B:04614', 'Mamu-B:04615', 'Mamu-B:04616', 'Mamu-B:04617',
#                             'Mamu-B:04701', 'Mamu-B:04702', 'Mamu-B:04703', 'Mamu-B:04704', 'Mamu-B:04705', 'Mamu-B:04801',
#                             'Mamu-B:04802', 'Mamu-B:04901', 'Mamu-B:05002', 'Mamu-B:05101', 'Mamu-B:05102', 'Mamu-B:05103',
#                             'Mamu-B:05104', 'Mamu-B:05105', 'Mamu-B:05201', 'Mamu-B:05301', 'Mamu-B:05302', 'Mamu-B:05401',
#                             'Mamu-B:05501', 'Mamu-B:05601', 'Mamu-B:05602', 'Mamu-B:05701', 'Mamu-B:05702', 'Mamu-B:05802',
#                             'Mamu-B:05901', 'Mamu-B:06001', 'Mamu-B:06002', 'Mamu-B:06003', 'Mamu-B:06101', 'Mamu-B:06102',
#                             'Mamu-B:06103', 'Mamu-B:06301', 'Mamu-B:06302', 'Mamu-B:06401', 'Mamu-B:06402', 'Mamu-B:06501',
#                             'Mamu-B:06502', 'Mamu-B:06503', 'Mamu-B:06601', 'Mamu-B:06701', 'Mamu-B:06702', 'Mamu-B:06801',
#                             'Mamu-B:06802', 'Mamu-B:06803', 'Mamu-B:06804', 'Mamu-B:06805', 'Mamu-B:06901', 'Mamu-B:06902',
#                             'Mamu-B:06903', 'Mamu-B:06904', 'Mamu-B:07001', 'Mamu-B:07002', 'Mamu-B:07101', 'Mamu-B:07201',
#                             'Mamu-B:07202', 'Mamu-B:07301', 'Mamu-B:07401', 'Mamu-B:07402', 'Mamu-B:07501', 'Mamu-B:07502',
#                             'Mamu-B:07601', 'Mamu-B:07602', 'Mamu-B:07701', 'Mamu-B:07702', 'Mamu-B:07801', 'Mamu-B:07901',
#                             'Mamu-B:07902', 'Mamu-B:07903', 'Mamu-B:08001', 'Mamu-B:08101', 'Mamu-B:08102', 'Mamu-B:08201',
#                             'Mamu-B:08202', 'Mamu-B:08301', 'Mamu-B:08401', 'Mamu-B:08501', 'Mamu-B:08502', 'Mamu-B:08601',
#                             'Mamu-B:08602', 'Mamu-B:08603', 'Mamu-B:08701', 'Mamu-B:08801', 'Mamu-B:08901', 'Mamu-B:09001',
#                             'Mamu-B:09101', 'Mamu-B:09102', 'Mamu-B:09201', 'Mamu-B:09301', 'Mamu-B:09401', 'Mamu-B:09501',
#                             'Mamu-B:09601', 'Mamu-B:09701', 'Mamu-B:09801', 'Mamu-B:09901', 'Mamu-B:10001', 'Mamu-B:10101',
#                             'Patr-A0101', 'Patr-A0201', 'Patr-A0301', 'Patr-A0302', 'Patr-A0401', 'Patr-A0402',
#                             'Patr-A0404', 'Patr-A0501', 'Patr-A0601', 'Patr-A0602', 'Patr-A0701', 'Patr-A0801',
#                             'Patr-A0802', 'Patr-A0803', 'Patr-A0901', 'Patr-A0902', 'Patr-A1001', 'Patr-A1101',
#                             'Patr-A1201', 'Patr-A1301', 'Patr-A1401', 'Patr-A1501', 'Patr-A1502', 'Patr-A1601',
#                             'Patr-A1701', 'Patr-A1702', 'Patr-A1703', 'Patr-A1801', 'Patr-A2301', 'Patr-A2401',
#                             'Patr-B0101', 'Patr-B0102', 'Patr-B0201', 'Patr-B0203', 'Patr-B0301', 'Patr-B0302',
#                             'Patr-B0401', 'Patr-B0402', 'Patr-B0501', 'Patr-B0502', 'Patr-B0601', 'Patr-B0701',
#                             'Patr-B0702', 'Patr-B0801', 'Patr-B0802', 'Patr-B0901', 'Patr-B1001', 'Patr-B1101',
#                             'Patr-B1102', 'Patr-B1202', 'Patr-B1301', 'Patr-B1401', 'Patr-B1601', 'Patr-B1602',
#                             'Patr-B1701', 'Patr-B1702', 'Patr-B1703', 'Patr-B1801', 'Patr-B1901', 'Patr-B2001',
#                             'Patr-B2101', 'Patr-B2201', 'Patr-B2202', 'Patr-B2301', 'Patr-B2302', 'Patr-B2303',
#                             'Patr-B2401', 'Patr-B2402', 'Patr-B2501', 'Patr-B2601', 'Patr-B2701', 'Patr-B2801',
#                             'Patr-B2901', 'Patr-B3001', 'Patr-B3501', 'Patr-B3601', 'Patr-B3701', 'Patr-C0201',
#                             'Patr-C0202', 'Patr-C0203', 'Patr-C0204', 'Patr-C0205', 'Patr-C0206', 'Patr-C0301',
#                             'Patr-C0302', 'Patr-C0303', 'Patr-C0304', 'Patr-C0401', 'Patr-C0501', 'Patr-C0502',
#                             'Patr-C0601', 'Patr-C0701', 'Patr-C0801', 'Patr-C0901', 'Patr-C0902', 'Patr-C0903',
#                             'Patr-C0904', 'Patr-C0905', 'Patr-C1001', 'Patr-C1101', 'Patr-C1201', 'Patr-C1301',
#                             'Patr-C1302', 'Patr-C1501', 'Patr-C1601', 'SLA-1:0101', 'SLA-1:0201', 'SLA-1:0202',
#                             'SLA-1:0401', 'SLA-1:0501', 'SLA-1:0601', 'SLA-1:0701', 'SLA-1:0702', 'SLA-1:0801',
#                             'SLA-1:1101', 'SLA-1:1201', 'SLA-1:1301', 'SLA-2:0101', 'SLA-2:0102', 'SLA-2:0201',
#                             'SLA-2:0202', 'SLA-2:0301', 'SLA-2:0302', 'SLA-2:0401', 'SLA-2:0402', 'SLA-2:0501',
#                             'SLA-2:0502', 'SLA-2:0601', 'SLA-2:0701', 'SLA-2:1001', 'SLA-2:1002', 'SLA-2:1101',
#                             'SLA-2:1201', 'SLA-3:0101', 'SLA-3:0301', 'SLA-3:0302', 'SLA-3:0303', 'SLA-3:0304',
#                             'SLA-3:0401', 'SLA-3:0501', 'SLA-3:0502', 'SLA-3:0503', 'SLA-3:0601', 'SLA-3:0602',
#                             'SLA-3:0701', 'SLA-6:0101', 'SLA-6:0102', 'SLA-6:0103', 'SLA-6:0104', 'SLA-6:0105']
#
#                             #better read  from  MHCs.txt if that exists in the original package or just read from
#                             #call  with -A or -listMHC
# # with open('/tmp/pan','rw') as f:
# #     c = 0
# #     for p in panalleles:
# #             c+=1
# #             f.write('\''+p+'\',')
# #             if c == 6:
# #                     f.write('\n')
# #                     c = 0
#
#
#     def make_predictions(self, peptides, alleles=None, method='netMHC-3.0', ignore=True):
#         if not alleles:
#             return
#         else:
#             assert all(isinstance(a, Allele) for a in alleles), "No list of Allele"
#         assert all(isinstance(a, AASequence) for a in peptides), "No list of AASequence"
#         # TODO: refactor these 3 lines for new peptide format:
#         peptides.sort(key=len)  # SUPERIMPORTANT for the afterwards use of groupby
#         pepset = Core.uniquify_list(peptides, Core.fred2_attrgetter('seq'))
#         pepsets = [list(g) for k, g in groupby(pepset, key=len)]
#
#         for ps in pepsets:
#             tmp_file = NamedTemporaryFile(delete=True)
    #             IO.write_peptide_file(ps, tmp_file.name)
#
#             length = len(ps[0].seq)
#
#             for allele in alleles:
#                 try:
#                     # logging.warning(allele)
#                     a = allele.to_netmhc(self, method)
#                 except LookupError:
#                     logging.warning("Allele not available for this method ("+method+"): "+str(allele))
#                     if ignore:
#                         continue
#                 if method == 'netMHC-3.0':
#                     cmd = self.netmhc_path + ' -a %s -p %s -l %s' % (a, tmp_file.name, length)
#                 elif method == 'netMHCpan-2.4':
#                     cmd = self.netpan_path + ' -a %s -p %s -l %s ' % (a, tmp_file.name, length)
#                 else:
#                     logging.warning('no such netMHC method known.')
#                 result = subprocess.check_output(cmd, shell=True)
#
#                 netsplit = [x.lstrip().split() for x in result.split('\n')[11:-3]] if method == 'netMHC-3.0' \
#                     else [x.lstrip().split() for x in result.split('\n')[57:-6]]
#                 #logging.warning(netsplit[0:3])
#
#                 rd = dict()
#                 if method == 'netMHC-3.0':
#                     for i in netsplit:
#                         if len(i) == len(self.mhcheader):
#                             rd[i[1]] = dict(zip(self.mhcheader, i))
#                         else:
#                             rd[i[1]] = dict(zip(self.mhcheader[:4]+self.mhcheader[5:], i))
#                 else:
#                     for i in netsplit:
#                             rd[i[2]] = dict(zip(self.panheader, i))  # here no matter if i is shorter than header
#
#                 for p in peptides:
#                     if str(p.seq) in rd:
#                         p.scores.append(Score(method, allele, rd[str(p.seq)]['logscore'], rd[str(p.seq)]['affinity(nM)'], None))

# This code is part of the Fred2 distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
__author__ = 'walzer', 'schubert'


import abc, collections, itertools, warnings, pandas, subprocess
from Fred2.Core.Result import Result
from Fred2.Core.Base import AEpitopePrediction
from tempfile import NamedTemporaryFile

class ANetMHC(AEpitopePrediction):
    """
        Abstract class representing NetMHC prediction function. These are wrapper of external binaries


    """

    @abc.abstractproperty
    def externalPath(self):
        """
        Specifies the external path of the executable

        :return: str - the path to the executable
        """
        raise NotImplementedError


    @abc.abstractmethod
    def parse_external_result(self, _file):
        """
        Parses external NetMHC results and returns a Result object

        :param str _file: The file path or the external prediction results
        :return: Result - Returns a Result object
        """
        raise NotImplementedError

    def predict(self, peptides, alleles=None, **kwargs):

        if isinstance(peptides, collections.Iterable):
            pep_seqs = {str(p):p for p in peptides}
        else:
            pep_seqs = {str(peptides):peptides}

        if alleles is None:
            allales_string = {conv_a:a for conv_a, a in itertools.izip(self.convert_alleles(self.supportedAlleles),
                                                                       self.supportedAlleles)}
        else:
            allales_string ={conv_a:a.name for conv_a, a in itertools.izip(self.convert_alleles(alleles),alleles)}

        result = None

        #group alleles in blocks of 80 alleles (NetMHC can't deal with more)
        allele_groups = []
        c_a = 0
        allele_group = []
        for a in allales_string.iterkeys():
            if c_a >= 80:
                c_a = 0
                allele_groups.append(allele_group)
                allele_group = [a]
            else:
                allele_group.append(a)
                c_a += 1

        #export peptides to peptide list
        tmp_file = NamedTemporaryFile(delete=True)
        with open(tmp_file.name, "w") as pep_file:
            pep_file.write("\n".join(pep_seqs.keys()))

        #generate cmd command
        cmd = self.externalPath + " -p %s -a %s"
        for allele_group in allele_groups:
            res = self.parse_external_result(
                subprocess.check_output(cmd%(tmp_file, ",".join(allele_group)), shell=True)
            )
            if result is None:
                result = res
            else:
                result_a, res_a = result.align(res, fill_value=0)
                result = result_a+res_a
        return result


class NetMHC(ANetMHC):
    """
        Implements the NetMHC binding (in current form for netMHC3.0)
        Possibility could exist for function injection to support also older versions

    """

    __alleles = ['A*24:02', 'A*24:03', 'B*53:01', 'B*27:05', 'A*23:01', 'A*02:04', 'A*29:02', 'A*02:06', 'A*02:01',
                 'A*02:02', 'A*02:03', 'A*26:02', 'A*26:01', 'A*31:01', 'B*07:02', 'A*68:01', 'A*68:02', 'B*35:01',
                 'B*58:01', 'B*57:01', 'B*15:01', 'A*69:01', 'B*54:01', 'A*11:01', 'A*03:01', 'B*40:01', 'B*40:02',
                 'B*44:02', 'A*30:01', 'A*02:19', 'A*30:02', 'B*39:01', 'A*02:16', 'B*51:01', 'B*45:01', 'A*02:12',
                 'A*02:11', 'B*08:01', 'B*18:01', 'B*44:03', 'B*08:02', 'A*33:01', 'A*01:01']
    __length = [9]
    __name = "netmhc"
    __externalPath = "/after/moon/thousendmiles/right/netMHC"

    def convert_alleles(self, alleles):
        return ["%s%s%s"%(a.locus, a.supertype, a.subtype) for a in alleles]

    def supportedAlleles(self):
        return self.__alleles

    def name(self):
        return self.__name

    def externalPath(self):
        return self.__externalPath

    def supportedLength(self):
        return self.__length

    def parse_external_result(self, _file):
        pass

    def predict(self, peptides, alleles=None, **kwargs):
        return super(NetMHC, self).predict(peptides, alleles=alleles, **kwargs)


class NetMHCpan(ANetMHC):
    """
        Implements the NetMHC binding (in current form for netMHCpan 2.4)
        Possibility could exist for function injection to support also older versions

        Supported  MHC alleles currently only restricted to HLA alleles
    """
    __length = [9]
    __name = "netmhcpan"
    __externalPath = "/after/moon/thousendmiles/right/netMHCpan"
    __alleles = ['A*01:01', 'A*01:02', 'A*01:03', 'A*01:06', 'A*01:07', 'A*01:08', 'A*01:09', 'A*01:10', 'A*01:12',
                 'A*01:13', 'A*01:14', 'A*01:17', 'A*01:19', 'A*01:20', 'A*01:21', 'A*01:23', 'A*01:24', 'A*01:25',
                 'A*01:26', 'A*01:28', 'A*01:29', 'A*01:30', 'A*01:32', 'A*01:33', 'A*01:35', 'A*01:36', 'A*01:37',
                 'A*01:38', 'A*01:39', 'A*01:40', 'A*01:41', 'A*01:42', 'A*01:43', 'A*01:44', 'A*01:45', 'A*01:46',
                 'A*01:47', 'A*01:48', 'A*01:49', 'A*01:50', 'A*01:51', 'A*01:54', 'A*01:55', 'A*01:58', 'A*01:59',
                 'A*01:60', 'A*01:61', 'A*01:62', 'A*01:63', 'A*01:64', 'A*01:65', 'A*01:66', 'A*02:01', 'A*02:02',
                 'A*02:03', 'A*02:04', 'A*02:05', 'A*02:06', 'A*02:07', 'A*02:08', 'A*02:09', 'A*02:10', 'A*02:101',
                 'A*02:102', 'A*02:103', 'A*02:104', 'A*02:105', 'A*02:106', 'A*02:107', 'A*02:108', 'A*02:109',
                 'A*02:11', 'A*02:110', 'A*02:111', 'A*02:112', 'A*02:114', 'A*02:115', 'A*02:116', 'A*02:117',
                 'A*02:118', 'A*02:119', 'A*02:12', 'A*02:120', 'A*02:121', 'A*02:122', 'A*02:123', 'A*02:124',
                 'A*02:126', 'A*02:127', 'A*02:128', 'A*02:129', 'A*02:13', 'A*02:130', 'A*02:131', 'A*02:132',
                 'A*02:133', 'A*02:134', 'A*02:135', 'A*02:136', 'A*02:137', 'A*02:138', 'A*02:139', 'A*02:14',
                 'A*02:140', 'A*02:141', 'A*02:142', 'A*02:143', 'A*02:144', 'A*02:145', 'A*02:146', 'A*02:147',
                 'A*02:148', 'A*02:149', 'A*02:150', 'A*02:151', 'A*02:152', 'A*02:153', 'A*02:154', 'A*02:155',
                 'A*02:156', 'A*02:157', 'A*02:158', 'A*02:159', 'A*02:16', 'A*02:160', 'A*02:161', 'A*02:162',
                 'A*02:163', 'A*02:164', 'A*02:165', 'A*02:166', 'A*02:167', 'A*02:168', 'A*02:169', 'A*02:17',
                 'A*02:170', 'A*02:171', 'A*02:172', 'A*02:173', 'A*02:174', 'A*02:175', 'A*02:176', 'A*02:177',
                 'A*02:178', 'A*02:179', 'A*02:18', 'A*02:180', 'A*02:181', 'A*02:182', 'A*02:183', 'A*02:184',
                 'A*02:185', 'A*02:186', 'A*02:187', 'A*02:188', 'A*02:189', 'A*02:19', 'A*02:190', 'A*02:191',
                 'A*02:192', 'A*02:193', 'A*02:194', 'A*02:195', 'A*02:196', 'A*02:197', 'A*02:198', 'A*02:199',
                 'A*02:20', 'A*02:200', 'A*02:201', 'A*02:202', 'A*02:203', 'A*02:204', 'A*02:205', 'A*02:206',
                 'A*02:207', 'A*02:208', 'A*02:209', 'A*02:21', 'A*02:210', 'A*02:211', 'A*02:212', 'A*02:213',
                 'A*02:214', 'A*02:215', 'A*02:216', 'A*02:217', 'A*02:218', 'A*02:219', 'A*02:22', 'A*02:220',
                 'A*02:221', 'A*02:224', 'A*02:228', 'A*02:229', 'A*02:230', 'A*02:231', 'A*02:232', 'A*02:233',
                 'A*02:234', 'A*02:235', 'A*02:236', 'A*02:237', 'A*02:238', 'A*02:239', 'A*02:24', 'A*02:240',
                 'A*02:241', 'A*02:242', 'A*02:243', 'A*02:244', 'A*02:245', 'A*02:246', 'A*02:247', 'A*02:248',
                 'A*02:249', 'A*02:25', 'A*02:251', 'A*02:252', 'A*02:253', 'A*02:254', 'A*02:255', 'A*02:256',
                 'A*02:257', 'A*02:258', 'A*02:259', 'A*02:26', 'A*02:260', 'A*02:261', 'A*02:262', 'A*02:263',
                 'A*02:264', 'A*02:265', 'A*02:266', 'A*02:27', 'A*02:28', 'A*02:29', 'A*02:30', 'A*02:31', 'A*02:33',
                 'A*02:34', 'A*02:35', 'A*02:36', 'A*02:37', 'A*02:38', 'A*02:39', 'A*02:40', 'A*02:41', 'A*02:42',
                 'A*02:44', 'A*02:45', 'A*02:46', 'A*02:47', 'A*02:48', 'A*02:49', 'A*02:50', 'A*02:51', 'A*02:52',
                 'A*02:54', 'A*02:55', 'A*02:56', 'A*02:57', 'A*02:58', 'A*02:59', 'A*02:60', 'A*02:61', 'A*02:62',
                 'A*02:63', 'A*02:64', 'A*02:65', 'A*02:66', 'A*02:67', 'A*02:68', 'A*02:69', 'A*02:70', 'A*02:71',
                 'A*02:72', 'A*02:73', 'A*02:74', 'A*02:75', 'A*02:76', 'A*02:77', 'A*02:78', 'A*02:79', 'A*02:80',
                 'A*02:81', 'A*02:84', 'A*02:85', 'A*02:86', 'A*02:87', 'A*02:89', 'A*02:90', 'A*02:91', 'A*02:92',
                 'A*02:93', 'A*02:95', 'A*02:96', 'A*02:97', 'A*02:99', 'A*03:01', 'A*03:02', 'A*03:04', 'A*03:05',
                 'A*03:06', 'A*03:07', 'A*03:08', 'A*03:09', 'A*03:10', 'A*03:12', 'A*03:13', 'A*03:14', 'A*03:15',
                 'A*03:16', 'A*03:17', 'A*03:18', 'A*03:19', 'A*03:20', 'A*03:22', 'A*03:23', 'A*03:24', 'A*03:25',
                 'A*03:26', 'A*03:27', 'A*03:28', 'A*03:29', 'A*03:30', 'A*03:31', 'A*03:32', 'A*03:33', 'A*03:34',
                 'A*03:35', 'A*03:37', 'A*03:38', 'A*03:39', 'A*03:40', 'A*03:41', 'A*03:42', 'A*03:43', 'A*03:44',
                 'A*03:45', 'A*03:46', 'A*03:47', 'A*03:48', 'A*03:49', 'A*03:50', 'A*03:51', 'A*03:52', 'A*03:53',
                 'A*03:54', 'A*03:55', 'A*03:56', 'A*03:57', 'A*03:58', 'A*03:59', 'A*03:60', 'A*03:61', 'A*03:62',
                 'A*03:63', 'A*03:64', 'A*03:65', 'A*03:66', 'A*03:67', 'A*03:70', 'A*03:71', 'A*03:72', 'A*03:73',
                 'A*03:74', 'A*03:75', 'A*03:76', 'A*03:77', 'A*03:78', 'A*03:79', 'A*03:80', 'A*03:81', 'A*03:82',
                 'A*11:01', 'A*11:02', 'A*11:03', 'A*11:04', 'A*11:05', 'A*11:06', 'A*11:07', 'A*11:08', 'A*11:09',
                 'A*11:10', 'A*11:11', 'A*11:12', 'A*11:13', 'A*11:14', 'A*11:15', 'A*11:16', 'A*11:17', 'A*11:18',
                 'A*11:19', 'A*11:20', 'A*11:22', 'A*11:23', 'A*11:24', 'A*11:25', 'A*11:26', 'A*11:27', 'A*11:29',
                 'A*11:30', 'A*11:31', 'A*11:32', 'A*11:33', 'A*11:34', 'A*11:35', 'A*11:36', 'A*11:37', 'A*11:38',
                 'A*11:39', 'A*11:40', 'A*11:41', 'A*11:42', 'A*11:43', 'A*11:44', 'A*11:45', 'A*11:46', 'A*11:47',
                 'A*11:48', 'A*11:49', 'A*11:51', 'A*11:53', 'A*11:54', 'A*11:55', 'A*11:56', 'A*11:57', 'A*11:58',
                 'A*11:59', 'A*11:60', 'A*11:61', 'A*11:62', 'A*11:63', 'A*11:64', 'A*23:01', 'A*23:02', 'A*23:03',
                 'A*23:04', 'A*23:05', 'A*23:06', 'A*23:09', 'A*23:10', 'A*23:12', 'A*23:13', 'A*23:14', 'A*23:15',
                 'A*23:16', 'A*23:17', 'A*23:18', 'A*23:20', 'A*23:21', 'A*23:22', 'A*23:23', 'A*23:24', 'A*23:25',
                 'A*23:26', 'A*24:02', 'A*24:03', 'A*24:04', 'A*24:05', 'A*24:06', 'A*24:07', 'A*24:08', 'A*24:10',
                 'A*24:100', 'A*24:101', 'A*24:102', 'A*24:103', 'A*24:104', 'A*24:105', 'A*24:106', 'A*24:107',
                 'A*24:108', 'A*24:109', 'A*24:110', 'A*24:111', 'A*24:112', 'A*24:113', 'A*24:114', 'A*24:115',
                 'A*24:116', 'A*24:117', 'A*24:118', 'A*24:119', 'A*24:120', 'A*24:121', 'A*24:122', 'A*24:123',
                 'A*24:124', 'A*24:125', 'A*24:126', 'A*24:127', 'A*24:128', 'A*24:129', 'A*24:13', 'A*24:130',
                 'A*24:131', 'A*24:133', 'A*24:134', 'A*24:135', 'A*24:136', 'A*24:137', 'A*24:138', 'A*24:139',
                 'A*24:14', 'A*24:140', 'A*24:141', 'A*24:142', 'A*24:143', 'A*24:144', 'A*24:15', 'A*24:17', 'A*24:18',
                 'A*24:19', 'A*24:20', 'A*24:21', 'A*24:22', 'A*24:23', 'A*24:24', 'A*24:25', 'A*24:26', 'A*24:27',
                 'A*24:28', 'A*24:29', 'A*24:30', 'A*24:31', 'A*24:32', 'A*24:33', 'A*24:34', 'A*24:35', 'A*24:37',
                 'A*24:38', 'A*24:39', 'A*24:41', 'A*24:42', 'A*24:43', 'A*24:44', 'A*24:46', 'A*24:47', 'A*24:49',
                 'A*24:50', 'A*24:51', 'A*24:52', 'A*24:53', 'A*24:54', 'A*24:55', 'A*24:56', 'A*24:57', 'A*24:58',
                 'A*24:59', 'A*24:61', 'A*24:62', 'A*24:63', 'A*24:64', 'A*24:66', 'A*24:67', 'A*24:68', 'A*24:69',
                 'A*24:70', 'A*24:71', 'A*24:72', 'A*24:73', 'A*24:74', 'A*24:75', 'A*24:76', 'A*24:77', 'A*24:78',
                 'A*24:79', 'A*24:80', 'A*24:81', 'A*24:82', 'A*24:85', 'A*24:87', 'A*24:88', 'A*24:89', 'A*24:91',
                 'A*24:92', 'A*24:93', 'A*24:94', 'A*24:95', 'A*24:96', 'A*24:97', 'A*24:98', 'A*24:99', 'A*25:01',
                 'A*25:02', 'A*25:03', 'A*25:04', 'A*25:05', 'A*25:06', 'A*25:07', 'A*25:08', 'A*25:09', 'A*25:10',
                 'A*25:11', 'A*25:13', 'A*26:01', 'A*26:02', 'A*26:03', 'A*26:04', 'A*26:05', 'A*26:06', 'A*26:07',
                 'A*26:08', 'A*26:09', 'A*26:10', 'A*26:12', 'A*26:13', 'A*26:14', 'A*26:15', 'A*26:16', 'A*26:17',
                 'A*26:18', 'A*26:19', 'A*26:20', 'A*26:21', 'A*26:22', 'A*26:23', 'A*26:24', 'A*26:26', 'A*26:27',
                 'A*26:28', 'A*26:29', 'A*26:30', 'A*26:31', 'A*26:32', 'A*26:33', 'A*26:34', 'A*26:35', 'A*26:36',
                 'A*26:37', 'A*26:38', 'A*26:39', 'A*26:40', 'A*26:41', 'A*26:42', 'A*26:43', 'A*26:45', 'A*26:46',
                 'A*26:47', 'A*26:48', 'A*26:49', 'A*26:50', 'A*29:01', 'A*29:02', 'A*29:03', 'A*29:04', 'A*29:05',
                 'A*29:06', 'A*29:07', 'A*29:09', 'A*29:10', 'A*29:11', 'A*29:12', 'A*29:13', 'A*29:14', 'A*29:15',
                 'A*29:16', 'A*29:17', 'A*29:18', 'A*29:19', 'A*29:20', 'A*29:21', 'A*29:22', 'A*30:01', 'A*30:02',
                 'A*30:03', 'A*30:04', 'A*30:06', 'A*30:07', 'A*30:08', 'A*30:09', 'A*30:10', 'A*30:11', 'A*30:12',
                 'A*30:13', 'A*30:15', 'A*30:16', 'A*30:17', 'A*30:18', 'A*30:19', 'A*30:20', 'A*30:22', 'A*30:23',
                 'A*30:24', 'A*30:25', 'A*30:26', 'A*30:28', 'A*30:29', 'A*30:30', 'A*30:31', 'A*30:32', 'A*30:33',
                 'A*30:34', 'A*30:35', 'A*30:36', 'A*30:37', 'A*30:38', 'A*30:39', 'A*30:40', 'A*30:41', 'A*31:01',
                 'A*31:02', 'A*31:03', 'A*31:04', 'A*31:05', 'A*31:06', 'A*31:07', 'A*31:08', 'A*31:09', 'A*31:10',
                 'A*31:11', 'A*31:12', 'A*31:13', 'A*31:15', 'A*31:16', 'A*31:17', 'A*31:18', 'A*31:19', 'A*31:20',
                 'A*31:21', 'A*31:22', 'A*31:23', 'A*31:24', 'A*31:25', 'A*31:26', 'A*31:27', 'A*31:28', 'A*31:29',
                 'A*31:30', 'A*31:31', 'A*31:32', 'A*31:33', 'A*31:34', 'A*31:35', 'A*31:36', 'A*31:37', 'A*32:01',
                 'A*32:02', 'A*32:03', 'A*32:04', 'A*32:05', 'A*32:06', 'A*32:07', 'A*32:08', 'A*32:09', 'A*32:10',
                 'A*32:12', 'A*32:13', 'A*32:14', 'A*32:15', 'A*32:16', 'A*32:17', 'A*32:18', 'A*32:20', 'A*32:21',
                 'A*32:22', 'A*32:23', 'A*32:24', 'A*32:25', 'A*33:01', 'A*33:03', 'A*33:04', 'A*33:05', 'A*33:06',
                 'A*33:07', 'A*33:08', 'A*33:09', 'A*33:10', 'A*33:11', 'A*33:12', 'A*33:13', 'A*33:14', 'A*33:15',
                 'A*33:16', 'A*33:17', 'A*33:18', 'A*33:19', 'A*33:20', 'A*33:21', 'A*33:22', 'A*33:23', 'A*33:24',
                 'A*33:25', 'A*33:26', 'A*33:27', 'A*33:28', 'A*33:29', 'A*33:30', 'A*33:31', 'A*34:01', 'A*34:02',
                 'A*34:03', 'A*34:04', 'A*34:05', 'A*34:06', 'A*34:07', 'A*34:08', 'A*36:01', 'A*36:02', 'A*36:03',
                 'A*36:04', 'A*36:05', 'A*43:01', 'A*66:01', 'A*66:02', 'A*66:03', 'A*66:04', 'A*66:05', 'A*66:06',
                 'A*66:07', 'A*66:08', 'A*66:09', 'A*66:10', 'A*66:11', 'A*66:12', 'A*66:13', 'A*66:14', 'A*66:15',
                 'A*68:01', 'A*68:02', 'A*68:03', 'A*68:04', 'A*68:05', 'A*68:06', 'A*68:07', 'A*68:08', 'A*68:09',
                 'A*68:10', 'A*68:12', 'A*68:13', 'A*68:14', 'A*68:15', 'A*68:16', 'A*68:17', 'A*68:19', 'A*68:20',
                 'A*68:21', 'A*68:22', 'A*68:23', 'A*68:24', 'A*68:25', 'A*68:26', 'A*68:27', 'A*68:28', 'A*68:29',
                 'A*68:30', 'A*68:31', 'A*68:32', 'A*68:33', 'A*68:34', 'A*68:35', 'A*68:36', 'A*68:37', 'A*68:38',
                 'A*68:39', 'A*68:40', 'A*68:41', 'A*68:42', 'A*68:43', 'A*68:44', 'A*68:45', 'A*68:46', 'A*68:47',
                 'A*68:48', 'A*68:50', 'A*68:51', 'A*68:52', 'A*68:53', 'A*68:54', 'A*69:01', 'A*74:01', 'A*74:02',
                 'A*74:03', 'A*74:04', 'A*74:05', 'A*74:06', 'A*74:07', 'A*74:08', 'A*74:09', 'A*74:10', 'A*74:11',
                 'A*74:13', 'A*80:01', 'A*80:02', 'B*07:02', 'B*07:03', 'B*07:04', 'B*07:05', 'B*07:06', 'B*07:07',
                 'B*07:08', 'B*07:09', 'B*07:10', 'B*07:100', 'B*07:101', 'B*07:102', 'B*07:103', 'B*07:104',
                 'B*07:105', 'B*07:106', 'B*07:107', 'B*07:108', 'B*07:109', 'B*07:11', 'B*07:110', 'B*07:112',
                 'B*07:113', 'B*07:114', 'B*07:115', 'B*07:12', 'B*07:13', 'B*07:14', 'B*07:15', 'B*07:16', 'B*07:17',
                 'B*07:18', 'B*07:19', 'B*07:20', 'B*07:21', 'B*07:22', 'B*07:23', 'B*07:24', 'B*07:25', 'B*07:26',
                 'B*07:27', 'B*07:28', 'B*07:29', 'B*07:30', 'B*07:31', 'B*07:32', 'B*07:33', 'B*07:34', 'B*07:35',
                 'B*07:36', 'B*07:37', 'B*07:38', 'B*07:39', 'B*07:40', 'B*07:41', 'B*07:42', 'B*07:43', 'B*07:44',
                 'B*07:45', 'B*07:46', 'B*07:47', 'B*07:48', 'B*07:50', 'B*07:51', 'B*07:52', 'B*07:53', 'B*07:54',
                 'B*07:55', 'B*07:56', 'B*07:57', 'B*07:58', 'B*07:59', 'B*07:60', 'B*07:61', 'B*07:62', 'B*07:63',
                 'B*07:64', 'B*07:65', 'B*07:66', 'B*07:68', 'B*07:69', 'B*07:70', 'B*07:71', 'B*07:72', 'B*07:73',
                 'B*07:74', 'B*07:75', 'B*07:76', 'B*07:77', 'B*07:78', 'B*07:79', 'B*07:80', 'B*07:81', 'B*07:82',
                 'B*07:83', 'B*07:84', 'B*07:85', 'B*07:86', 'B*07:87', 'B*07:88', 'B*07:89', 'B*07:90', 'B*07:91',
                 'B*07:92', 'B*07:93', 'B*07:94', 'B*07:95', 'B*07:96', 'B*07:97', 'B*07:98', 'B*07:99', 'B*08:01',
                 'B*08:02', 'B*08:03', 'B*08:04', 'B*08:05', 'B*08:07', 'B*08:09', 'B*08:10', 'B*08:11', 'B*08:12',
                 'B*08:13', 'B*08:14', 'B*08:15', 'B*08:16', 'B*08:17', 'B*08:18', 'B*08:20', 'B*08:21', 'B*08:22',
                 'B*08:23', 'B*08:24', 'B*08:25', 'B*08:26', 'B*08:27', 'B*08:28', 'B*08:29', 'B*08:31', 'B*08:32',
                 'B*08:33', 'B*08:34', 'B*08:35', 'B*08:36', 'B*08:37', 'B*08:38', 'B*08:39', 'B*08:40', 'B*08:41',
                 'B*08:42', 'B*08:43', 'B*08:44', 'B*08:45', 'B*08:46', 'B*08:47', 'B*08:48', 'B*08:49', 'B*08:50',
                 'B*08:51', 'B*08:52', 'B*08:53', 'B*08:54', 'B*08:55', 'B*08:56', 'B*08:57', 'B*08:58', 'B*08:59',
                 'B*08:60', 'B*08:61', 'B*08:62', 'B*13:01', 'B*13:02', 'B*13:03', 'B*13:04', 'B*13:06', 'B*13:09',
                 'B*13:10', 'B*13:11', 'B*13:12', 'B*13:13', 'B*13:14', 'B*13:15', 'B*13:16', 'B*13:17', 'B*13:18',
                 'B*13:19', 'B*13:20', 'B*13:21', 'B*13:22', 'B*13:23', 'B*13:25', 'B*13:26', 'B*13:27', 'B*13:28',
                 'B*13:29', 'B*13:30', 'B*13:31', 'B*13:32', 'B*13:33', 'B*13:34', 'B*13:35', 'B*13:36', 'B*13:37',
                 'B*13:38', 'B*13:39', 'B*14:01', 'B*14:02', 'B*14:03', 'B*14:04', 'B*14:05', 'B*14:06', 'B*14:08',
                 'B*14:09', 'B*14:10', 'B*14:11', 'B*14:12', 'B*14:13', 'B*14:14', 'B*14:15', 'B*14:16', 'B*14:17',
                 'B*14:18', 'B*15:01', 'B*15:02', 'B*15:03', 'B*15:04', 'B*15:05', 'B*15:06', 'B*15:07', 'B*15:08',
                 'B*15:09', 'B*15:10', 'B*15:101', 'B*15:102', 'B*15:103', 'B*15:104', 'B*15:105', 'B*15:106',
                 'B*15:107', 'B*15:108', 'B*15:109', 'B*15:11', 'B*15:110', 'B*15:112', 'B*15:113', 'B*15:114',
                 'B*15:115', 'B*15:116', 'B*15:117', 'B*15:118', 'B*15:119', 'B*15:12', 'B*15:120', 'B*15:121',
                 'B*15:122', 'B*15:123', 'B*15:124', 'B*15:125', 'B*15:126', 'B*15:127', 'B*15:128', 'B*15:129',
                 'B*15:13', 'B*15:131', 'B*15:132', 'B*15:133', 'B*15:134', 'B*15:135', 'B*15:136', 'B*15:137',
                 'B*15:138', 'B*15:139', 'B*15:14', 'B*15:140', 'B*15:141', 'B*15:142', 'B*15:143', 'B*15:144',
                 'B*15:145', 'B*15:146', 'B*15:147', 'B*15:148', 'B*15:15', 'B*15:150', 'B*15:151', 'B*15:152',
                 'B*15:153', 'B*15:154', 'B*15:155', 'B*15:156', 'B*15:157', 'B*15:158', 'B*15:159', 'B*15:16',
                 'B*15:160', 'B*15:161', 'B*15:162', 'B*15:163', 'B*15:164', 'B*15:165', 'B*15:166', 'B*15:167',
                 'B*15:168', 'B*15:169', 'B*15:17', 'B*15:170', 'B*15:171', 'B*15:172', 'B*15:173', 'B*15:174',
                 'B*15:175', 'B*15:176', 'B*15:177', 'B*15:178', 'B*15:179', 'B*15:18', 'B*15:180', 'B*15:183',
                 'B*15:184', 'B*15:185', 'B*15:186', 'B*15:187', 'B*15:188', 'B*15:189', 'B*15:19', 'B*15:191',
                 'B*15:192', 'B*15:193', 'B*15:194', 'B*15:195', 'B*15:196', 'B*15:197', 'B*15:198', 'B*15:199',
                 'B*15:20', 'B*15:200', 'B*15:201', 'B*15:202', 'B*15:21', 'B*15:23', 'B*15:24', 'B*15:25', 'B*15:27',
                 'B*15:28', 'B*15:29', 'B*15:30', 'B*15:31', 'B*15:32', 'B*15:33', 'B*15:34', 'B*15:35', 'B*15:36',
                 'B*15:37', 'B*15:38', 'B*15:39', 'B*15:40', 'B*15:42', 'B*15:43', 'B*15:44', 'B*15:45', 'B*15:46',
                 'B*15:47', 'B*15:48', 'B*15:49', 'B*15:50', 'B*15:51', 'B*15:52', 'B*15:53', 'B*15:54', 'B*15:55',
                 'B*15:56', 'B*15:57', 'B*15:58', 'B*15:60', 'B*15:61', 'B*15:62', 'B*15:63', 'B*15:64', 'B*15:65',
                 'B*15:66', 'B*15:67', 'B*15:68', 'B*15:69', 'B*15:70', 'B*15:71', 'B*15:72', 'B*15:73', 'B*15:74',
                 'B*15:75', 'B*15:76', 'B*15:77', 'B*15:78', 'B*15:80', 'B*15:81', 'B*15:82', 'B*15:83', 'B*15:84',
                 'B*15:85', 'B*15:86', 'B*15:87', 'B*15:88', 'B*15:89', 'B*15:90', 'B*15:91', 'B*15:92', 'B*15:93',
                 'B*15:95', 'B*15:96', 'B*15:97', 'B*15:98', 'B*15:99', 'B*18:01', 'B*18:02', 'B*18:03', 'B*18:04',
                 'B*18:05', 'B*18:06', 'B*18:07', 'B*18:08', 'B*18:09', 'B*18:10', 'B*18:11', 'B*18:12', 'B*18:13',
                 'B*18:14', 'B*18:15', 'B*18:18', 'B*18:19', 'B*18:20', 'B*18:21', 'B*18:22', 'B*18:24', 'B*18:25',
                 'B*18:26', 'B*18:27', 'B*18:28', 'B*18:29', 'B*18:30', 'B*18:31', 'B*18:32', 'B*18:33', 'B*18:34',
                 'B*18:35', 'B*18:36', 'B*18:37', 'B*18:38', 'B*18:39', 'B*18:40', 'B*18:41', 'B*18:42', 'B*18:43',
                 'B*18:44', 'B*18:45', 'B*18:46', 'B*18:47', 'B*18:48', 'B*18:49', 'B*18:50', 'B*27:01', 'B*27:02',
                 'B*27:03', 'B*27:04', 'B*27:05', 'B*27:06', 'B*27:07', 'B*27:08', 'B*27:09', 'B*27:10', 'B*27:11',
                 'B*27:12', 'B*27:13', 'B*27:14', 'B*27:15', 'B*27:16', 'B*27:17', 'B*27:18', 'B*27:19', 'B*27:20',
                 'B*27:21', 'B*27:23', 'B*27:24', 'B*27:25', 'B*27:26', 'B*27:27', 'B*27:28', 'B*27:29', 'B*27:30',
                 'B*27:31', 'B*27:32', 'B*27:33', 'B*27:34', 'B*27:35', 'B*27:36', 'B*27:37', 'B*27:38', 'B*27:39',
                 'B*27:40', 'B*27:41', 'B*27:42', 'B*27:43', 'B*27:44', 'B*27:45', 'B*27:46', 'B*27:47', 'B*27:48',
                 'B*27:49', 'B*27:50', 'B*27:51', 'B*27:52', 'B*27:53', 'B*27:54', 'B*27:55', 'B*27:56', 'B*27:57',
                 'B*27:58', 'B*27:60', 'B*27:61', 'B*27:62', 'B*27:63', 'B*27:67', 'B*27:68', 'B*27:69', 'B*35:01',
                 'B*35:02', 'B*35:03', 'B*35:04', 'B*35:05', 'B*35:06', 'B*35:07', 'B*35:08', 'B*35:09', 'B*35:10',
                 'B*35:100', 'B*35:101', 'B*35:102', 'B*35:103', 'B*35:104', 'B*35:105', 'B*35:106', 'B*35:107',
                 'B*35:108', 'B*35:109', 'B*35:11', 'B*35:110', 'B*35:111', 'B*35:112', 'B*35:113', 'B*35:114',
                 'B*35:115', 'B*35:116', 'B*35:117', 'B*35:118', 'B*35:119', 'B*35:12', 'B*35:120', 'B*35:121',
                 'B*35:122', 'B*35:123', 'B*35:124', 'B*35:125', 'B*35:126', 'B*35:127', 'B*35:128', 'B*35:13',
                 'B*35:131', 'B*35:132', 'B*35:133', 'B*35:135', 'B*35:136', 'B*35:137', 'B*35:138', 'B*35:139',
                 'B*35:14', 'B*35:140', 'B*35:141', 'B*35:142', 'B*35:143', 'B*35:144', 'B*35:15', 'B*35:16', 'B*35:17',
                 'B*35:18', 'B*35:19', 'B*35:20', 'B*35:21', 'B*35:22', 'B*35:23', 'B*35:24', 'B*35:25', 'B*35:26',
                 'B*35:27', 'B*35:28', 'B*35:29', 'B*35:30', 'B*35:31', 'B*35:32', 'B*35:33', 'B*35:34', 'B*35:35',
                 'B*35:36', 'B*35:37', 'B*35:38', 'B*35:39', 'B*35:41', 'B*35:42', 'B*35:43', 'B*35:44', 'B*35:45',
                 'B*35:46', 'B*35:47', 'B*35:48', 'B*35:49', 'B*35:50', 'B*35:51', 'B*35:52', 'B*35:54', 'B*35:55',
                 'B*35:56', 'B*35:57', 'B*35:58', 'B*35:59', 'B*35:60', 'B*35:61', 'B*35:62', 'B*35:63', 'B*35:64',
                 'B*35:66', 'B*35:67', 'B*35:68', 'B*35:69', 'B*35:70', 'B*35:71', 'B*35:72', 'B*35:74', 'B*35:75',
                 'B*35:76', 'B*35:77', 'B*35:78', 'B*35:79', 'B*35:80', 'B*35:81', 'B*35:82', 'B*35:83', 'B*35:84',
                 'B*35:85', 'B*35:86', 'B*35:87', 'B*35:88', 'B*35:89', 'B*35:90', 'B*35:91', 'B*35:92', 'B*35:93',
                 'B*35:94', 'B*35:95', 'B*35:96', 'B*35:97', 'B*35:98', 'B*35:99', 'B*37:01', 'B*37:02', 'B*37:04',
                 'B*37:05', 'B*37:06', 'B*37:07', 'B*37:08', 'B*37:09', 'B*37:10', 'B*37:11', 'B*37:12', 'B*37:13',
                 'B*37:14', 'B*37:15', 'B*37:17', 'B*37:18', 'B*37:19', 'B*37:20', 'B*37:21', 'B*37:22', 'B*37:23',
                 'B*38:01', 'B*38:02', 'B*38:03', 'B*38:04', 'B*38:05', 'B*38:06', 'B*38:07', 'B*38:08', 'B*38:09',
                 'B*38:10', 'B*38:11', 'B*38:12', 'B*38:13', 'B*38:14', 'B*38:15', 'B*38:16', 'B*38:17', 'B*38:18',
                 'B*38:19', 'B*38:20', 'B*38:21', 'B*38:22', 'B*38:23', 'B*39:01', 'B*39:02', 'B*39:03', 'B*39:04',
                 'B*39:05', 'B*39:06', 'B*39:07', 'B*39:08', 'B*39:09', 'B*39:10', 'B*39:11', 'B*39:12', 'B*39:13',
                 'B*39:14', 'B*39:15', 'B*39:16', 'B*39:17', 'B*39:18', 'B*39:19', 'B*39:20', 'B*39:22', 'B*39:23',
                 'B*39:24', 'B*39:26', 'B*39:27', 'B*39:28', 'B*39:29', 'B*39:30', 'B*39:31', 'B*39:32', 'B*39:33',
                 'B*39:34', 'B*39:35', 'B*39:36', 'B*39:37', 'B*39:39', 'B*39:41', 'B*39:42', 'B*39:43', 'B*39:44',
                 'B*39:45', 'B*39:46', 'B*39:47', 'B*39:48', 'B*39:49', 'B*39:50', 'B*39:51', 'B*39:52', 'B*39:53',
                 'B*39:54', 'B*39:55', 'B*39:56', 'B*39:57', 'B*39:58', 'B*39:59', 'B*39:60', 'B*40:01', 'B*40:02',
                 'B*40:03', 'B*40:04', 'B*40:05', 'B*40:06', 'B*40:07', 'B*40:08', 'B*40:09', 'B*40:10', 'B*40:100',
                 'B*40:101', 'B*40:102', 'B*40:103', 'B*40:104', 'B*40:105', 'B*40:106', 'B*40:107', 'B*40:108',
                 'B*40:109', 'B*40:11', 'B*40:110', 'B*40:111', 'B*40:112', 'B*40:113', 'B*40:114', 'B*40:115',
                 'B*40:116', 'B*40:117', 'B*40:119', 'B*40:12', 'B*40:120', 'B*40:121', 'B*40:122', 'B*40:123',
                 'B*40:124', 'B*40:125', 'B*40:126', 'B*40:127', 'B*40:128', 'B*40:129', 'B*40:13', 'B*40:130',
                 'B*40:131', 'B*40:132', 'B*40:134', 'B*40:135', 'B*40:136', 'B*40:137', 'B*40:138', 'B*40:139',
                 'B*40:14', 'B*40:140', 'B*40:141', 'B*40:143', 'B*40:145', 'B*40:146', 'B*40:147', 'B*40:15',
                 'B*40:16', 'B*40:18', 'B*40:19', 'B*40:20', 'B*40:21', 'B*40:23', 'B*40:24', 'B*40:25', 'B*40:26',
                 'B*40:27', 'B*40:28', 'B*40:29', 'B*40:30', 'B*40:31', 'B*40:32', 'B*40:33', 'B*40:34', 'B*40:35',
                 'B*40:36', 'B*40:37', 'B*40:38', 'B*40:39', 'B*40:40', 'B*40:42', 'B*40:43', 'B*40:44', 'B*40:45',
                 'B*40:46', 'B*40:47', 'B*40:48', 'B*40:49', 'B*40:50', 'B*40:51', 'B*40:52', 'B*40:53', 'B*40:54',
                 'B*40:55', 'B*40:56', 'B*40:57', 'B*40:58', 'B*40:59', 'B*40:60', 'B*40:61', 'B*40:62', 'B*40:63',
                 'B*40:64', 'B*40:65', 'B*40:66', 'B*40:67', 'B*40:68', 'B*40:69', 'B*40:70', 'B*40:71', 'B*40:72',
                 'B*40:73', 'B*40:74', 'B*40:75', 'B*40:76', 'B*40:77', 'B*40:78', 'B*40:79', 'B*40:80', 'B*40:81',
                 'B*40:82', 'B*40:83', 'B*40:84', 'B*40:85', 'B*40:86', 'B*40:87', 'B*40:88', 'B*40:89', 'B*40:90',
                 'B*40:91', 'B*40:92', 'B*40:93', 'B*40:94', 'B*40:95', 'B*40:96', 'B*40:97', 'B*40:98', 'B*40:99',
                 'B*41:01', 'B*41:02', 'B*41:03', 'B*41:04', 'B*41:05', 'B*41:06', 'B*41:07', 'B*41:08', 'B*41:09',
                 'B*41:10', 'B*41:11', 'B*41:12', 'B*42:01', 'B*42:02', 'B*42:04', 'B*42:05', 'B*42:06', 'B*42:07',
                 'B*42:08', 'B*42:09', 'B*42:10', 'B*42:11', 'B*42:12', 'B*42:13', 'B*42:14', 'B*44:02', 'B*44:03',
                 'B*44:04', 'B*44:05', 'B*44:06', 'B*44:07', 'B*44:08', 'B*44:09', 'B*44:10', 'B*44:100', 'B*44:101',
                 'B*44:102', 'B*44:103', 'B*44:104', 'B*44:105', 'B*44:106', 'B*44:107', 'B*44:109', 'B*44:11',
                 'B*44:110', 'B*44:12', 'B*44:13', 'B*44:14', 'B*44:15', 'B*44:16', 'B*44:17', 'B*44:18', 'B*44:20',
                 'B*44:21', 'B*44:22', 'B*44:24', 'B*44:25', 'B*44:26', 'B*44:27', 'B*44:28', 'B*44:29', 'B*44:30',
                 'B*44:31', 'B*44:32', 'B*44:33', 'B*44:34', 'B*44:35', 'B*44:36', 'B*44:37', 'B*44:38', 'B*44:39',
                 'B*44:40', 'B*44:41', 'B*44:42', 'B*44:43', 'B*44:44', 'B*44:45', 'B*44:46', 'B*44:47', 'B*44:48',
                 'B*44:49', 'B*44:50', 'B*44:51', 'B*44:53', 'B*44:54', 'B*44:55', 'B*44:57', 'B*44:59', 'B*44:60',
                 'B*44:62', 'B*44:63', 'B*44:64', 'B*44:65', 'B*44:66', 'B*44:67', 'B*44:68', 'B*44:69', 'B*44:70',
                 'B*44:71', 'B*44:72', 'B*44:73', 'B*44:74', 'B*44:75', 'B*44:76', 'B*44:77', 'B*44:78', 'B*44:79',
                 'B*44:80', 'B*44:81', 'B*44:82', 'B*44:83', 'B*44:84', 'B*44:85', 'B*44:86', 'B*44:87', 'B*44:88',
                 'B*44:89', 'B*44:90', 'B*44:91', 'B*44:92', 'B*44:93', 'B*44:94', 'B*44:95', 'B*44:96', 'B*44:97',
                 'B*44:98', 'B*44:99', 'B*45:01', 'B*45:02', 'B*45:03', 'B*45:04', 'B*45:05', 'B*45:06', 'B*45:07',
                 'B*45:08', 'B*45:09', 'B*45:10', 'B*45:11', 'B*45:12', 'B*46:01', 'B*46:02', 'B*46:03', 'B*46:04',
                 'B*46:05', 'B*46:06', 'B*46:08', 'B*46:09', 'B*46:10', 'B*46:11', 'B*46:12', 'B*46:13', 'B*46:14',
                 'B*46:16', 'B*46:17', 'B*46:18', 'B*46:19', 'B*46:20', 'B*46:21', 'B*46:22', 'B*46:23', 'B*46:24',
                 'B*47:01', 'B*47:02', 'B*47:03', 'B*47:04', 'B*47:05', 'B*47:06', 'B*47:07', 'B*48:01', 'B*48:02',
                 'B*48:03', 'B*48:04', 'B*48:05', 'B*48:06', 'B*48:07', 'B*48:08', 'B*48:09', 'B*48:10', 'B*48:11',
                 'B*48:12', 'B*48:13', 'B*48:14', 'B*48:15', 'B*48:16', 'B*48:17', 'B*48:18', 'B*48:19', 'B*48:20',
                 'B*48:21', 'B*48:22', 'B*48:23', 'B*49:01', 'B*49:02', 'B*49:03', 'B*49:04', 'B*49:05', 'B*49:06',
                 'B*49:07', 'B*49:08', 'B*49:09', 'B*49:10', 'B*50:01', 'B*50:02', 'B*50:04', 'B*50:05', 'B*50:06',
                 'B*50:07', 'B*50:08', 'B*50:09', 'B*51:01', 'B*51:02', 'B*51:03', 'B*51:04', 'B*51:05', 'B*51:06',
                 'B*51:07', 'B*51:08', 'B*51:09', 'B*51:12', 'B*51:13', 'B*51:14', 'B*51:15', 'B*51:16', 'B*51:17',
                 'B*51:18', 'B*51:19', 'B*51:20', 'B*51:21', 'B*51:22', 'B*51:23', 'B*51:24', 'B*51:26', 'B*51:28',
                 'B*51:29', 'B*51:30', 'B*51:31', 'B*51:32', 'B*51:33', 'B*51:34', 'B*51:35', 'B*51:36', 'B*51:37',
                 'B*51:38', 'B*51:39', 'B*51:40', 'B*51:42', 'B*51:43', 'B*51:45', 'B*51:46', 'B*51:48', 'B*51:49',
                 'B*51:50', 'B*51:51', 'B*51:52', 'B*51:53', 'B*51:54', 'B*51:55', 'B*51:56', 'B*51:57', 'B*51:58',
                 'B*51:59', 'B*51:60', 'B*51:61', 'B*51:62', 'B*51:63', 'B*51:64', 'B*51:65', 'B*51:66', 'B*51:67',
                 'B*51:68', 'B*51:69', 'B*51:70', 'B*51:71', 'B*51:72', 'B*51:73', 'B*51:74', 'B*51:75', 'B*51:76',
                 'B*51:77', 'B*51:78', 'B*51:79', 'B*51:80', 'B*51:81', 'B*51:82', 'B*51:83', 'B*51:84', 'B*51:85',
                 'B*51:86', 'B*51:87', 'B*51:88', 'B*51:89', 'B*51:90', 'B*51:91', 'B*51:92', 'B*51:93', 'B*51:94',
                 'B*51:95', 'B*51:96', 'B*52:01', 'B*52:02', 'B*52:03', 'B*52:04', 'B*52:05', 'B*52:06', 'B*52:07',
                 'B*52:08', 'B*52:09', 'B*52:10', 'B*52:11', 'B*52:12', 'B*52:13', 'B*52:14', 'B*52:15', 'B*52:16',
                 'B*52:17', 'B*52:18', 'B*52:19', 'B*52:20', 'B*52:21', 'B*53:01', 'B*53:02', 'B*53:03', 'B*53:04',
                 'B*53:05', 'B*53:06', 'B*53:07', 'B*53:08', 'B*53:09', 'B*53:10', 'B*53:11', 'B*53:12', 'B*53:13',
                 'B*53:14', 'B*53:15', 'B*53:16', 'B*53:17', 'B*53:18', 'B*53:19', 'B*53:20', 'B*53:21', 'B*53:22',
                 'B*53:23', 'B*54:01', 'B*54:02', 'B*54:03', 'B*54:04', 'B*54:06', 'B*54:07', 'B*54:09', 'B*54:10',
                 'B*54:11', 'B*54:12', 'B*54:13', 'B*54:14', 'B*54:15', 'B*54:16', 'B*54:17', 'B*54:18', 'B*54:19',
                 'B*54:20', 'B*54:21', 'B*54:22', 'B*54:23', 'B*55:01', 'B*55:02', 'B*55:03', 'B*55:04', 'B*55:05',
                 'B*55:07', 'B*55:08', 'B*55:09', 'B*55:10', 'B*55:11', 'B*55:12', 'B*55:13', 'B*55:14', 'B*55:15',
                 'B*55:16', 'B*55:17', 'B*55:18', 'B*55:19', 'B*55:20', 'B*55:21', 'B*55:22', 'B*55:23', 'B*55:24',
                 'B*55:25', 'B*55:26', 'B*55:27', 'B*55:28', 'B*55:29', 'B*55:30', 'B*55:31', 'B*55:32', 'B*55:33',
                 'B*55:34', 'B*55:35', 'B*55:36', 'B*55:37', 'B*55:38', 'B*55:39', 'B*55:40', 'B*55:41', 'B*55:42',
                 'B*55:43', 'B*56:01', 'B*56:02', 'B*56:03', 'B*56:04', 'B*56:05', 'B*56:06', 'B*56:07', 'B*56:08',
                 'B*56:09', 'B*56:10', 'B*56:11', 'B*56:12', 'B*56:13', 'B*56:14', 'B*56:15', 'B*56:16', 'B*56:17',
                 'B*56:18', 'B*56:20', 'B*56:21', 'B*56:22', 'B*56:23', 'B*56:24', 'B*56:25', 'B*56:26', 'B*56:27',
                 'B*56:29', 'B*57:01', 'B*57:02', 'B*57:03', 'B*57:04', 'B*57:05', 'B*57:06', 'B*57:07', 'B*57:08',
                 'B*57:09', 'B*57:10', 'B*57:11', 'B*57:12', 'B*57:13', 'B*57:14', 'B*57:15', 'B*57:16', 'B*57:17',
                 'B*57:18', 'B*57:19', 'B*57:20', 'B*57:21', 'B*57:22', 'B*57:23', 'B*57:24', 'B*57:25', 'B*57:26',
                 'B*57:27', 'B*57:29', 'B*57:30', 'B*57:31', 'B*57:32', 'B*58:01', 'B*58:02', 'B*58:04', 'B*58:05',
                 'B*58:06', 'B*58:07', 'B*58:08', 'B*58:09', 'B*58:11', 'B*58:12', 'B*58:13', 'B*58:14', 'B*58:15',
                 'B*58:16', 'B*58:18', 'B*58:19', 'B*58:20', 'B*58:21', 'B*58:22', 'B*58:23', 'B*58:24', 'B*58:25',
                 'B*58:26', 'B*58:27', 'B*58:28', 'B*58:29', 'B*58:30', 'B*59:01', 'B*59:02', 'B*59:03', 'B*59:04',
                 'B*59:05', 'B*67:01', 'B*67:02', 'B*73:01', 'B*73:02', 'B*78:01', 'B*78:02', 'B*78:03', 'B*78:04',
                 'B*78:05', 'B*78:06', 'B*78:07', 'B*81:01', 'B*81:02', 'B*81:03', 'B*81:05', 'B*82:01', 'B*82:02',
                 'B*82:03', 'B*83:01', 'C*01:02', 'C*01:03', 'C*01:04', 'C*01:05', 'C*01:06', 'C*01:07', 'C*01:08',
                 'C*01:09', 'C*01:10', 'C*01:11', 'C*01:12', 'C*01:13', 'C*01:14', 'C*01:15', 'C*01:16', 'C*01:17',
                 'C*01:18', 'C*01:19', 'C*01:20', 'C*01:21', 'C*01:22', 'C*01:23', 'C*01:24', 'C*01:25', 'C*01:26',
                 'C*01:27', 'C*01:28', 'C*01:29', 'C*01:30', 'C*01:31', 'C*01:32', 'C*01:33', 'C*01:34', 'C*01:35',
                 'C*01:36', 'C*01:38', 'C*01:39', 'C*01:40', 'C*02:02', 'C*02:03', 'C*02:04', 'C*02:05', 'C*02:06',
                 'C*02:07', 'C*02:08', 'C*02:09', 'C*02:10', 'C*02:11', 'C*02:12', 'C*02:13', 'C*02:14', 'C*02:15',
                 'C*02:16', 'C*02:17', 'C*02:18', 'C*02:19', 'C*02:20', 'C*02:21', 'C*02:22', 'C*02:23', 'C*02:24',
                 'C*02:26', 'C*02:27', 'C*02:28', 'C*02:29', 'C*02:30', 'C*02:31', 'C*02:32', 'C*02:33', 'C*02:34',
                 'C*02:35', 'C*02:36', 'C*02:37', 'C*02:39', 'C*02:40', 'C*03:01', 'C*03:02', 'C*03:03', 'C*03:04',
                 'C*03:05', 'C*03:06', 'C*03:07', 'C*03:08', 'C*03:09', 'C*03:10', 'C*03:11', 'C*03:12', 'C*03:13',
                 'C*03:14', 'C*03:15', 'C*03:16', 'C*03:17', 'C*03:18', 'C*03:19', 'C*03:21', 'C*03:23', 'C*03:24',
                 'C*03:25', 'C*03:26', 'C*03:27', 'C*03:28', 'C*03:29', 'C*03:30', 'C*03:31', 'C*03:32', 'C*03:33',
                 'C*03:34', 'C*03:35', 'C*03:36', 'C*03:37', 'C*03:38', 'C*03:39', 'C*03:40', 'C*03:41', 'C*03:42',
                 'C*03:43', 'C*03:44', 'C*03:45', 'C*03:46', 'C*03:47', 'C*03:48', 'C*03:49', 'C*03:50', 'C*03:51',
                 'C*03:52', 'C*03:53', 'C*03:54', 'C*03:55', 'C*03:56', 'C*03:57', 'C*03:58', 'C*03:59', 'C*03:60',
                 'C*03:61', 'C*03:62', 'C*03:63', 'C*03:64', 'C*03:65', 'C*03:66', 'C*03:67', 'C*03:68', 'C*03:69',
                 'C*03:70', 'C*03:71', 'C*03:72', 'C*03:73', 'C*03:74', 'C*03:75', 'C*03:76', 'C*03:77', 'C*03:78',
                 'C*03:79', 'C*03:80', 'C*03:81', 'C*03:82', 'C*03:83', 'C*03:84', 'C*03:85', 'C*03:86', 'C*03:87',
                 'C*03:88', 'C*03:89', 'C*03:90', 'C*03:91', 'C*03:92', 'C*03:93', 'C*03:94', 'C*04:01', 'C*04:03',
                 'C*04:04', 'C*04:05', 'C*04:06', 'C*04:07', 'C*04:08', 'C*04:10', 'C*04:11', 'C*04:12', 'C*04:13',
                 'C*04:14', 'C*04:15', 'C*04:16', 'C*04:17', 'C*04:18', 'C*04:19', 'C*04:20', 'C*04:23', 'C*04:24',
                 'C*04:25', 'C*04:26', 'C*04:27', 'C*04:28', 'C*04:29', 'C*04:30', 'C*04:31', 'C*04:32', 'C*04:33',
                 'C*04:34', 'C*04:35', 'C*04:36', 'C*04:37', 'C*04:38', 'C*04:39', 'C*04:40', 'C*04:41', 'C*04:42',
                 'C*04:43', 'C*04:44', 'C*04:45', 'C*04:46', 'C*04:47', 'C*04:48', 'C*04:49', 'C*04:50', 'C*04:51',
                 'C*04:52', 'C*04:53', 'C*04:54', 'C*04:55', 'C*04:56', 'C*04:57', 'C*04:58', 'C*04:60', 'C*04:61',
                 'C*04:62', 'C*04:63', 'C*04:64', 'C*04:65', 'C*04:66', 'C*04:67', 'C*04:68', 'C*04:69', 'C*04:70',
                 'C*05:01', 'C*05:03', 'C*05:04', 'C*05:05', 'C*05:06', 'C*05:08', 'C*05:09', 'C*05:10', 'C*05:11',
                 'C*05:12', 'C*05:13', 'C*05:14', 'C*05:15', 'C*05:16', 'C*05:17', 'C*05:18', 'C*05:19', 'C*05:20',
                 'C*05:21', 'C*05:22', 'C*05:23', 'C*05:24', 'C*05:25', 'C*05:26', 'C*05:27', 'C*05:28', 'C*05:29',
                 'C*05:30', 'C*05:31', 'C*05:32', 'C*05:33', 'C*05:34', 'C*05:35', 'C*05:36', 'C*05:37', 'C*05:38',
                 'C*05:39', 'C*05:40', 'C*05:41', 'C*05:42', 'C*05:43', 'C*05:44', 'C*05:45', 'C*06:02', 'C*06:03',
                 'C*06:04', 'C*06:05', 'C*06:06', 'C*06:07', 'C*06:08', 'C*06:09', 'C*06:10', 'C*06:11', 'C*06:12',
                 'C*06:13', 'C*06:14', 'C*06:15', 'C*06:17', 'C*06:18', 'C*06:19', 'C*06:20', 'C*06:21', 'C*06:22',
                 'C*06:23', 'C*06:24', 'C*06:25', 'C*06:26', 'C*06:27', 'C*06:28', 'C*06:29', 'C*06:30', 'C*06:31',
                 'C*06:32', 'C*06:33', 'C*06:34', 'C*06:35', 'C*06:36', 'C*06:37', 'C*06:38', 'C*06:39', 'C*06:40',
                 'C*06:41', 'C*06:42', 'C*06:43', 'C*06:44', 'C*06:45', 'C*07:01', 'C*07:02', 'C*07:03', 'C*07:04',
                 'C*07:05', 'C*07:06', 'C*07:07', 'C*07:08', 'C*07:09', 'C*07:10', 'C*07:100', 'C*07:101', 'C*07:102',
                 'C*07:103', 'C*07:105', 'C*07:106', 'C*07:107', 'C*07:108', 'C*07:109', 'C*07:11', 'C*07:110',
                 'C*07:111', 'C*07:112', 'C*07:113', 'C*07:114', 'C*07:115', 'C*07:116', 'C*07:117', 'C*07:118',
                 'C*07:119', 'C*07:12', 'C*07:120', 'C*07:122', 'C*07:123', 'C*07:124', 'C*07:125', 'C*07:126',
                 'C*07:127', 'C*07:128', 'C*07:129', 'C*07:13', 'C*07:130', 'C*07:131', 'C*07:132', 'C*07:133',
                 'C*07:134', 'C*07:135', 'C*07:136', 'C*07:137', 'C*07:138', 'C*07:139', 'C*07:14', 'C*07:140',
                 'C*07:141', 'C*07:142', 'C*07:143', 'C*07:144', 'C*07:145', 'C*07:146', 'C*07:147', 'C*07:148',
                 'C*07:149', 'C*07:15', 'C*07:16', 'C*07:17', 'C*07:18', 'C*07:19', 'C*07:20', 'C*07:21', 'C*07:22',
                 'C*07:23', 'C*07:24', 'C*07:25', 'C*07:26', 'C*07:27', 'C*07:28', 'C*07:29', 'C*07:30', 'C*07:31',
                 'C*07:35', 'C*07:36', 'C*07:37', 'C*07:38', 'C*07:39', 'C*07:40', 'C*07:41', 'C*07:42', 'C*07:43',
                 'C*07:44', 'C*07:45', 'C*07:46', 'C*07:47', 'C*07:48', 'C*07:49', 'C*07:50', 'C*07:51', 'C*07:52',
                 'C*07:53', 'C*07:54', 'C*07:56', 'C*07:57', 'C*07:58', 'C*07:59', 'C*07:60', 'C*07:62', 'C*07:63',
                 'C*07:64', 'C*07:65', 'C*07:66', 'C*07:67', 'C*07:68', 'C*07:69', 'C*07:70', 'C*07:71', 'C*07:72',
                 'C*07:73', 'C*07:74', 'C*07:75', 'C*07:76', 'C*07:77', 'C*07:78', 'C*07:79', 'C*07:80', 'C*07:81',
                 'C*07:82', 'C*07:83', 'C*07:84', 'C*07:85', 'C*07:86', 'C*07:87', 'C*07:88', 'C*07:89', 'C*07:90',
                 'C*07:91', 'C*07:92', 'C*07:93', 'C*07:94', 'C*07:95', 'C*07:96', 'C*07:97', 'C*07:99', 'C*08:01',
                 'C*08:02', 'C*08:03', 'C*08:04', 'C*08:05', 'C*08:06', 'C*08:07', 'C*08:08', 'C*08:09', 'C*08:10',
                 'C*08:11', 'C*08:12', 'C*08:13', 'C*08:14', 'C*08:15', 'C*08:16', 'C*08:17', 'C*08:18', 'C*08:19',
                 'C*08:20', 'C*08:21', 'C*08:22', 'C*08:23', 'C*08:24', 'C*08:25', 'C*08:27', 'C*08:28', 'C*08:29',
                 'C*08:30', 'C*08:31', 'C*08:32', 'C*08:33', 'C*08:34', 'C*08:35', 'C*12:02', 'C*12:03', 'C*12:04',
                 'C*12:05', 'C*12:06', 'C*12:07', 'C*12:08', 'C*12:09', 'C*12:10', 'C*12:11', 'C*12:12', 'C*12:13',
                 'C*12:14', 'C*12:15', 'C*12:16', 'C*12:17', 'C*12:18', 'C*12:19', 'C*12:20', 'C*12:21', 'C*12:22',
                 'C*12:23', 'C*12:24', 'C*12:25', 'C*12:26', 'C*12:27', 'C*12:28', 'C*12:29', 'C*12:30', 'C*12:31',
                 'C*12:32', 'C*12:33', 'C*12:34', 'C*12:35', 'C*12:36', 'C*12:37', 'C*12:38', 'C*12:40', 'C*12:41',
                 'C*12:43', 'C*12:44', 'C*14:02', 'C*14:03', 'C*14:04', 'C*14:05', 'C*14:06', 'C*14:08', 'C*14:09',
                 'C*14:10', 'C*14:11', 'C*14:12', 'C*14:13', 'C*14:14', 'C*14:15', 'C*14:16', 'C*14:17', 'C*14:18',
                 'C*14:19', 'C*14:20', 'C*15:02', 'C*15:03', 'C*15:04', 'C*15:05', 'C*15:06', 'C*15:07', 'C*15:08',
                 'C*15:09', 'C*15:10', 'C*15:11', 'C*15:12', 'C*15:13', 'C*15:15', 'C*15:16', 'C*15:17', 'C*15:18',
                 'C*15:19', 'C*15:20', 'C*15:21', 'C*15:22', 'C*15:23', 'C*15:24', 'C*15:25', 'C*15:26', 'C*15:27',
                 'C*15:28', 'C*15:29', 'C*15:30', 'C*15:31', 'C*15:33', 'C*15:34', 'C*15:35', 'C*16:01', 'C*16:02',
                 'C*16:04', 'C*16:06', 'C*16:07', 'C*16:08', 'C*16:09', 'C*16:10', 'C*16:11', 'C*16:12', 'C*16:13',
                 'C*16:14', 'C*16:15', 'C*16:17', 'C*16:18', 'C*16:19', 'C*16:20', 'C*16:21', 'C*16:22', 'C*16:23',
                 'C*16:24', 'C*16:25', 'C*16:26', 'C*17:01', 'C*17:02', 'C*17:03', 'C*17:04', 'C*17:05', 'C*17:06',
                 'C*17:07', 'C*18:01', 'C*18:02', 'C*18:03', 'E*01:01', 'G*01:01', 'G*01:02', 'G*01:03', 'G*01:04',
                 'G*01:06', 'G*01:07', 'G*01:08', 'G*01:09']

    def convert_alleles(self, alleles):
        return ["HLA-%s%s:%s"%(a.locus, a.supertype, a.subtype) for a in alleles]

    def supportedAlleles(self):
        return self.__alleles

    def name(self):
        return self.__name

    def externalPath(self):
        return self.__externalPath

    def supportedLength(self):
        return self.__length