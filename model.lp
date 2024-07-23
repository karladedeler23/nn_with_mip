\ Model neural_network_training
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  hinge_loss_terms[0,0] + hinge_loss_terms[0,1] + hinge_loss_terms[0,2]
   + hinge_loss_terms[0,3] + hinge_loss_terms[0,4] + hinge_loss_terms[0,5]
   + hinge_loss_terms[0,6] + hinge_loss_terms[0,7] + hinge_loss_terms[0,8]
   + hinge_loss_terms[0,9] + hinge_loss_terms[1,0] + hinge_loss_terms[1,1]
   + hinge_loss_terms[1,2] + hinge_loss_terms[1,3] + hinge_loss_terms[1,4]
   + hinge_loss_terms[1,5] + hinge_loss_terms[1,6] + hinge_loss_terms[1,7]
   + hinge_loss_terms[1,8] + hinge_loss_terms[1,9] + hinge_loss_terms[2,0]
   + hinge_loss_terms[2,1] + hinge_loss_terms[2,2] + hinge_loss_terms[2,3]
   + hinge_loss_terms[2,4] + hinge_loss_terms[2,5] + hinge_loss_terms[2,6]
   + hinge_loss_terms[2,7] + hinge_loss_terms[2,8] + hinge_loss_terms[2,9]
   + hinge_loss_terms[3,0] + hinge_loss_terms[3,1] + hinge_loss_terms[3,2]
   + hinge_loss_terms[3,3] + hinge_loss_terms[3,4] + hinge_loss_terms[3,5]
   + hinge_loss_terms[3,6] + hinge_loss_terms[3,7] + hinge_loss_terms[3,8]
   + hinge_loss_terms[3,9] + 0.0001 abs_W1[0,0] + 0.0001 abs_W1[0,1]
   + 0.0001 abs_W1[0,2] + 0.0001 abs_W1[0,3] + 0.0001 abs_W1[0,4]
   + 0.0001 abs_W1[0,5] + 0.0001 abs_W1[0,6] + 0.0001 abs_W1[0,7]
   + 0.0001 abs_W1[0,8] + 0.0001 abs_W1[0,9] + 0.0001 abs_W1[0,10]
   + 0.0001 abs_W1[0,11] + 0.0001 abs_W1[0,12] + 0.0001 abs_W1[0,13]
   + 0.0001 abs_W1[0,14] + 0.0001 abs_W1[0,15] + 0.0001 abs_W1[1,0]
   + 0.0001 abs_W1[1,1] + 0.0001 abs_W1[1,2] + 0.0001 abs_W1[1,3]
   + 0.0001 abs_W1[1,4] + 0.0001 abs_W1[1,5] + 0.0001 abs_W1[1,6]
   + 0.0001 abs_W1[1,7] + 0.0001 abs_W1[1,8] + 0.0001 abs_W1[1,9]
   + 0.0001 abs_W1[1,10] + 0.0001 abs_W1[1,11] + 0.0001 abs_W1[1,12]
   + 0.0001 abs_W1[1,13] + 0.0001 abs_W1[1,14] + 0.0001 abs_W1[1,15]
   + 0.0001 abs_W1[2,0] + 0.0001 abs_W1[2,1] + 0.0001 abs_W1[2,2]
   + 0.0001 abs_W1[2,3] + 0.0001 abs_W1[2,4] + 0.0001 abs_W1[2,5]
   + 0.0001 abs_W1[2,6] + 0.0001 abs_W1[2,7] + 0.0001 abs_W1[2,8]
   + 0.0001 abs_W1[2,9] + 0.0001 abs_W1[2,10] + 0.0001 abs_W1[2,11]
   + 0.0001 abs_W1[2,12] + 0.0001 abs_W1[2,13] + 0.0001 abs_W1[2,14]
   + 0.0001 abs_W1[2,15] + 0.0001 abs_W1[3,0] + 0.0001 abs_W1[3,1]
   + 0.0001 abs_W1[3,2] + 0.0001 abs_W1[3,3] + 0.0001 abs_W1[3,4]
   + 0.0001 abs_W1[3,5] + 0.0001 abs_W1[3,6] + 0.0001 abs_W1[3,7]
   + 0.0001 abs_W1[3,8] + 0.0001 abs_W1[3,9] + 0.0001 abs_W1[3,10]
   + 0.0001 abs_W1[3,11] + 0.0001 abs_W1[3,12] + 0.0001 abs_W1[3,13]
   + 0.0001 abs_W1[3,14] + 0.0001 abs_W1[3,15] + 0.0001 abs_b1[0]
   + 0.0001 abs_b1[1] + 0.0001 abs_b1[2] + 0.0001 abs_b1[3]
   + 0.0001 abs_W2[0,0] + 0.0001 abs_W2[0,1] + 0.0001 abs_W2[0,2]
   + 0.0001 abs_W2[0,3] + 0.0001 abs_W2[1,0] + 0.0001 abs_W2[1,1]
   + 0.0001 abs_W2[1,2] + 0.0001 abs_W2[1,3] + 0.0001 abs_W2[2,0]
   + 0.0001 abs_W2[2,1] + 0.0001 abs_W2[2,2] + 0.0001 abs_W2[2,3]
   + 0.0001 abs_W2[3,0] + 0.0001 abs_W2[3,1] + 0.0001 abs_W2[3,2]
   + 0.0001 abs_W2[3,3] + 0.0001 abs_W2[4,0] + 0.0001 abs_W2[4,1]
   + 0.0001 abs_W2[4,2] + 0.0001 abs_W2[4,3] + 0.0001 abs_W2[5,0]
   + 0.0001 abs_W2[5,1] + 0.0001 abs_W2[5,2] + 0.0001 abs_W2[5,3]
   + 0.0001 abs_W2[6,0] + 0.0001 abs_W2[6,1] + 0.0001 abs_W2[6,2]
   + 0.0001 abs_W2[6,3] + 0.0001 abs_W2[7,0] + 0.0001 abs_W2[7,1]
   + 0.0001 abs_W2[7,2] + 0.0001 abs_W2[7,3] + 0.0001 abs_W2[8,0]
   + 0.0001 abs_W2[8,1] + 0.0001 abs_W2[8,2] + 0.0001 abs_W2[8,3]
   + 0.0001 abs_W2[9,0] + 0.0001 abs_W2[9,1] + 0.0001 abs_W2[9,2]
   + 0.0001 abs_W2[9,3] + 0.0001 abs_b2[0] + 0.0001 abs_b2[1]
   + 0.0001 abs_b2[2] + 0.0001 abs_b2[3] + 0.0001 abs_b2[4]
   + 0.0001 abs_b2[5] + 0.0001 abs_b2[6] + 0.0001 abs_b2[7]
   + 0.0001 abs_b2[8] + 0.0001 abs_b2[9]
Subject To
 R0: 1.12198472162998 W1[0,0] + 2.711375320771233 W1[0,1]
   + 1.447596895835376 W1[0,2] + 1.063768536858581 W1[0,3]
   + 1.141831072053693 W1[0,4] + 2.193362862024767 W1[0,5]
   - 0.3757789107455998 W1[0,6] + 1.454775431015482 W1[0,7]
   - 1.272935173363989 W1[0,8] - 0.325254096008859 W1[0,9]
   - 0.742899911641654 W1[0,10] - 2.29933459621834 W1[0,11]
   + 0.8274862354383623 W1[0,12] - 1.932466365663255 W1[0,13]
   + 1.127253055081355 W1[0,14] - 0.7940698923379773 W1[0,15] - b1[0]
   + z1[0,0] = 0
 R1: - z1[0,0] + a1=max(0,z)[0,0] >= 0
 R2: a1=max(0,z)[0,0] >= 0
 R3: - z1[0,0] + a1=max(0,z)[0,0] + 16 binary_vars1[0,0] <= 16
 R4: a1=max(0,z)[0,0] - 16 binary_vars1[0,0] <= 0
 R5: 1.12198472162998 W1[1,0] + 2.711375320771233 W1[1,1]
   + 1.447596895835376 W1[1,2] + 1.063768536858581 W1[1,3]
   + 1.141831072053693 W1[1,4] + 2.193362862024767 W1[1,5]
   - 0.3757789107455998 W1[1,6] + 1.454775431015482 W1[1,7]
   - 1.272935173363989 W1[1,8] - 0.325254096008859 W1[1,9]
   - 0.742899911641654 W1[1,10] - 2.29933459621834 W1[1,11]
   + 0.8274862354383623 W1[1,12] - 1.932466365663255 W1[1,13]
   + 1.127253055081355 W1[1,14] - 0.7940698923379773 W1[1,15] - b1[1]
   + z1[0,1] = 0
 R6: - z1[0,1] + a1=max(0,z)[0,1] >= 0
 R7: a1=max(0,z)[0,1] >= 0
 R8: - z1[0,1] + a1=max(0,z)[0,1] + 16 binary_vars1[0,1] <= 16
 R9: a1=max(0,z)[0,1] - 16 binary_vars1[0,1] <= 0
 R10: 1.12198472162998 W1[2,0] + 2.711375320771233 W1[2,1]
   + 1.447596895835376 W1[2,2] + 1.063768536858581 W1[2,3]
   + 1.141831072053693 W1[2,4] + 2.193362862024767 W1[2,5]
   - 0.3757789107455998 W1[2,6] + 1.454775431015482 W1[2,7]
   - 1.272935173363989 W1[2,8] - 0.325254096008859 W1[2,9]
   - 0.742899911641654 W1[2,10] - 2.29933459621834 W1[2,11]
   + 0.8274862354383623 W1[2,12] - 1.932466365663255 W1[2,13]
   + 1.127253055081355 W1[2,14] - 0.7940698923379773 W1[2,15] - b1[2]
   + z1[0,2] = 0
 R11: - z1[0,2] + a1=max(0,z)[0,2] >= 0
 R12: a1=max(0,z)[0,2] >= 0
 R13: - z1[0,2] + a1=max(0,z)[0,2] + 16 binary_vars1[0,2] <= 16
 R14: a1=max(0,z)[0,2] - 16 binary_vars1[0,2] <= 0
 R15: 1.12198472162998 W1[3,0] + 2.711375320771233 W1[3,1]
   + 1.447596895835376 W1[3,2] + 1.063768536858581 W1[3,3]
   + 1.141831072053693 W1[3,4] + 2.193362862024767 W1[3,5]
   - 0.3757789107455998 W1[3,6] + 1.454775431015482 W1[3,7]
   - 1.272935173363989 W1[3,8] - 0.325254096008859 W1[3,9]
   - 0.742899911641654 W1[3,10] - 2.29933459621834 W1[3,11]
   + 0.8274862354383623 W1[3,12] - 1.932466365663255 W1[3,13]
   + 1.127253055081355 W1[3,14] - 0.7940698923379773 W1[3,15] - b1[3]
   + z1[0,3] = 0
 R16: - z1[0,3] + a1=max(0,z)[0,3] >= 0
 R17: a1=max(0,z)[0,3] >= 0
 R18: - z1[0,3] + a1=max(0,z)[0,3] + 16 binary_vars1[0,3] <= 16
 R19: a1=max(0,z)[0,3] - 16 binary_vars1[0,3] <= 0
 R20: 1.12198472162998 W1[0,0] + 1.642954628586199 W1[0,1]
   + 0.3430148072290612 W1[0,2] + 0.7582271081966589 W1[0,3]
   - 0.6047752471861643 W1[0,4] - 0.9116016791391609 W1[0,5]
   - 1.58185383450843 W1[0,6] - 1.807076311805337 W1[0,7]
   - 0.5603057532519531 W1[0,8] - 1.490663704180214 W1[0,9]
   + 0.3075122021822665 W1[0,10] - 0.5534992039271948 W1[0,11]
   + 1.181495589116223 W1[0,12] + 0.3251150157306338 W1[0,13]
   + 0.741980323278689 W1[0,14] + 0.7977218335197896 W1[0,15] - b1[0]
   + z1[1,0] = 0
 R21: - z1[1,0] + a1=max(0,z)[1,0] >= 0
 R22: a1=max(0,z)[1,0] >= 0
 R23: - z1[1,0] + a1=max(0,z)[1,0] + 16 binary_vars1[1,0] <= 16
 R24: a1=max(0,z)[1,0] - 16 binary_vars1[1,0] <= 0
 R25: 1.12198472162998 W1[1,0] + 1.642954628586199 W1[1,1]
   + 0.3430148072290612 W1[1,2] + 0.7582271081966589 W1[1,3]
   - 0.6047752471861643 W1[1,4] - 0.9116016791391609 W1[1,5]
   - 1.58185383450843 W1[1,6] - 1.807076311805337 W1[1,7]
   - 0.5603057532519531 W1[1,8] - 1.490663704180214 W1[1,9]
   + 0.3075122021822665 W1[1,10] - 0.5534992039271948 W1[1,11]
   + 1.181495589116223 W1[1,12] + 0.3251150157306338 W1[1,13]
   + 0.741980323278689 W1[1,14] + 0.7977218335197896 W1[1,15] - b1[1]
   + z1[1,1] = 0
 R26: - z1[1,1] + a1=max(0,z)[1,1] >= 0
 R27: a1=max(0,z)[1,1] >= 0
 R28: - z1[1,1] + a1=max(0,z)[1,1] + 16 binary_vars1[1,1] <= 16
 R29: a1=max(0,z)[1,1] - 16 binary_vars1[1,1] <= 0
 R30: 1.12198472162998 W1[2,0] + 1.642954628586199 W1[2,1]
   + 0.3430148072290612 W1[2,2] + 0.7582271081966589 W1[2,3]
   - 0.6047752471861643 W1[2,4] - 0.9116016791391609 W1[2,5]
   - 1.58185383450843 W1[2,6] - 1.807076311805337 W1[2,7]
   - 0.5603057532519531 W1[2,8] - 1.490663704180214 W1[2,9]
   + 0.3075122021822665 W1[2,10] - 0.5534992039271948 W1[2,11]
   + 1.181495589116223 W1[2,12] + 0.3251150157306338 W1[2,13]
   + 0.741980323278689 W1[2,14] + 0.7977218335197896 W1[2,15] - b1[2]
   + z1[1,2] = 0
 R31: - z1[1,2] + a1=max(0,z)[1,2] >= 0
 R32: a1=max(0,z)[1,2] >= 0
 R33: - z1[1,2] + a1=max(0,z)[1,2] + 16 binary_vars1[1,2] <= 16
 R34: a1=max(0,z)[1,2] - 16 binary_vars1[1,2] <= 0
 R35: 1.12198472162998 W1[3,0] + 1.642954628586199 W1[3,1]
   + 0.3430148072290612 W1[3,2] + 0.7582271081966589 W1[3,3]
   - 0.6047752471861643 W1[3,4] - 0.9116016791391609 W1[3,5]
   - 1.58185383450843 W1[3,6] - 1.807076311805337 W1[3,7]
   - 0.5603057532519531 W1[3,8] - 1.490663704180214 W1[3,9]
   + 0.3075122021822665 W1[3,10] - 0.5534992039271948 W1[3,11]
   + 1.181495589116223 W1[3,12] + 0.3251150157306338 W1[3,13]
   + 0.741980323278689 W1[3,14] + 0.7977218335197896 W1[3,15] - b1[3]
   + z1[1,3] = 0
 R36: - z1[1,3] + a1=max(0,z)[1,3] >= 0
 R37: a1=max(0,z)[1,3] >= 0
 R38: - z1[1,3] + a1=max(0,z)[1,3] + 16 binary_vars1[1,3] <= 16
 R39: a1=max(0,z)[1,3] - 16 binary_vars1[1,3] <= 0
 R40: 1.12198472162998 W1[0,0] - 0.2564599352983068 W1[0,1]
   + 0.4953709573816563 W1[0,2] - 0.8713271780002586 W1[0,3]
   + 0.2542114671940932 W1[0,4] - 0.3636667601102325 W1[0,5]
   + 0.7325061543337574 W1[0,6] - 0.0130578532538864 W1[0,7]
   + 1.250960689532806 W1[0,8] + 0.6944793111410769 W1[0,9]
   + 0.6576495734569067 W1[0,10] + 1.301450900382148 W1[0,11]
   - 0.6328023484828134 W1[0,12] + 1.017439972691426 W1[0,13]
   - 1.280701518685305 W1[0,14] + 0.6301648097452879 W1[0,15] - b1[0]
   + z1[2,0] = 0
 R41: - z1[2,0] + a1=max(0,z)[2,0] >= 0
 R42: a1=max(0,z)[2,0] >= 0
 R43: - z1[2,0] + a1=max(0,z)[2,0] + 16 binary_vars1[2,0] <= 16
 R44: a1=max(0,z)[2,0] - 16 binary_vars1[2,0] <= 0
 R45: 1.12198472162998 W1[1,0] - 0.2564599352983068 W1[1,1]
   + 0.4953709573816563 W1[1,2] - 0.8713271780002586 W1[1,3]
   + 0.2542114671940932 W1[1,4] - 0.3636667601102325 W1[1,5]
   + 0.7325061543337574 W1[1,6] - 0.0130578532538864 W1[1,7]
   + 1.250960689532806 W1[1,8] + 0.6944793111410769 W1[1,9]
   + 0.6576495734569067 W1[1,10] + 1.301450900382148 W1[1,11]
   - 0.6328023484828134 W1[1,12] + 1.017439972691426 W1[1,13]
   - 1.280701518685305 W1[1,14] + 0.6301648097452879 W1[1,15] - b1[1]
   + z1[2,1] = 0
 R46: - z1[2,1] + a1=max(0,z)[2,1] >= 0
 R47: a1=max(0,z)[2,1] >= 0
 R48: - z1[2,1] + a1=max(0,z)[2,1] + 16 binary_vars1[2,1] <= 16
 R49: a1=max(0,z)[2,1] - 16 binary_vars1[2,1] <= 0
 R50: 1.12198472162998 W1[2,0] - 0.2564599352983068 W1[2,1]
   + 0.4953709573816563 W1[2,2] - 0.8713271780002586 W1[2,3]
   + 0.2542114671940932 W1[2,4] - 0.3636667601102325 W1[2,5]
   + 0.7325061543337574 W1[2,6] - 0.0130578532538864 W1[2,7]
   + 1.250960689532806 W1[2,8] + 0.6944793111410769 W1[2,9]
   + 0.6576495734569067 W1[2,10] + 1.301450900382148 W1[2,11]
   - 0.6328023484828134 W1[2,12] + 1.017439972691426 W1[2,13]
   - 1.280701518685305 W1[2,14] + 0.6301648097452879 W1[2,15] - b1[2]
   + z1[2,2] = 0
 R51: - z1[2,2] + a1=max(0,z)[2,2] >= 0
 R52: a1=max(0,z)[2,2] >= 0
 R53: - z1[2,2] + a1=max(0,z)[2,2] + 16 binary_vars1[2,2] <= 16
 R54: a1=max(0,z)[2,2] - 16 binary_vars1[2,2] <= 0
 R55: 1.12198472162998 W1[3,0] - 0.2564599352983068 W1[3,1]
   + 0.4953709573816563 W1[3,2] - 0.8713271780002586 W1[3,3]
   + 0.2542114671940932 W1[3,4] - 0.3636667601102325 W1[3,5]
   + 0.7325061543337574 W1[3,6] - 0.0130578532538864 W1[3,7]
   + 1.250960689532806 W1[3,8] + 0.6944793111410769 W1[3,9]
   + 0.6576495734569067 W1[3,10] + 1.301450900382148 W1[3,11]
   - 0.6328023484828134 W1[3,12] + 1.017439972691426 W1[3,13]
   - 1.280701518685305 W1[3,14] + 0.6301648097452879 W1[3,15] - b1[3]
   + z1[2,3] = 0
 R56: - z1[2,3] + a1=max(0,z)[2,3] >= 0
 R57: a1=max(0,z)[2,3] >= 0
 R58: - z1[2,3] + a1=max(0,z)[2,3] + 16 binary_vars1[2,3] <= 16
 R59: a1=max(0,z)[2,3] - 16 binary_vars1[2,3] <= 0
 R60: 0.071558274281243 W1[0,0] + 0.5151772312797736 W1[0,1]
   - 0.6473001687628067 W1[0,2] - 0.8713271780002586 W1[0,3]
   - 1.406496180607738 W1[0,4] - 0.984659668343018 W1[0,5]
   - 0.5387620085513877 W1[0,6] - 0.6980467192462584 W1[0,7]
   - 0.708770215775294 W1[0,8] - 0.1431588447320848 W1[0,9]
   - 0.5543644040322324 W1[0,10] + 0.9741067643275577 W1[0,11]
   + 1.004490912277293 W1[0,12] + 1.07764214286193 W1[0,13]
   + 1.127253055081355 W1[0,14] + 0.3229769328253679 W1[0,15] - b1[0]
   + z1[3,0] = 0
 R61: - z1[3,0] + a1=max(0,z)[3,0] >= 0
 R62: a1=max(0,z)[3,0] >= 0
 R63: - z1[3,0] + a1=max(0,z)[3,0] + 16 binary_vars1[3,0] <= 16
 R64: a1=max(0,z)[3,0] - 16 binary_vars1[3,0] <= 0
 R65: 0.071558274281243 W1[1,0] + 0.5151772312797736 W1[1,1]
   - 0.6473001687628067 W1[1,2] - 0.8713271780002586 W1[1,3]
   - 1.406496180607738 W1[1,4] - 0.984659668343018 W1[1,5]
   - 0.5387620085513877 W1[1,6] - 0.6980467192462584 W1[1,7]
   - 0.708770215775294 W1[1,8] - 0.1431588447320848 W1[1,9]
   - 0.5543644040322324 W1[1,10] + 0.9741067643275577 W1[1,11]
   + 1.004490912277293 W1[1,12] + 1.07764214286193 W1[1,13]
   + 1.127253055081355 W1[1,14] + 0.3229769328253679 W1[1,15] - b1[1]
   + z1[3,1] = 0
 R66: - z1[3,1] + a1=max(0,z)[3,1] >= 0
 R67: a1=max(0,z)[3,1] >= 0
 R68: - z1[3,1] + a1=max(0,z)[3,1] + 16 binary_vars1[3,1] <= 16
 R69: a1=max(0,z)[3,1] - 16 binary_vars1[3,1] <= 0
 R70: 0.071558274281243 W1[2,0] + 0.5151772312797736 W1[2,1]
   - 0.6473001687628067 W1[2,2] - 0.8713271780002586 W1[2,3]
   - 1.406496180607738 W1[2,4] - 0.984659668343018 W1[2,5]
   - 0.5387620085513877 W1[2,6] - 0.6980467192462584 W1[2,7]
   - 0.708770215775294 W1[2,8] - 0.1431588447320848 W1[2,9]
   - 0.5543644040322324 W1[2,10] + 0.9741067643275577 W1[2,11]
   + 1.004490912277293 W1[2,12] + 1.07764214286193 W1[2,13]
   + 1.127253055081355 W1[2,14] + 0.3229769328253679 W1[2,15] - b1[2]
   + z1[3,2] = 0
 R71: - z1[3,2] + a1=max(0,z)[3,2] >= 0
 R72: a1=max(0,z)[3,2] >= 0
 R73: - z1[3,2] + a1=max(0,z)[3,2] + 16 binary_vars1[3,2] <= 16
 R74: a1=max(0,z)[3,2] - 16 binary_vars1[3,2] <= 0
 R75: 0.071558274281243 W1[3,0] + 0.5151772312797736 W1[3,1]
   - 0.6473001687628067 W1[3,2] - 0.8713271780002586 W1[3,3]
   - 1.406496180607738 W1[3,4] - 0.984659668343018 W1[3,5]
   - 0.5387620085513877 W1[3,6] - 0.6980467192462584 W1[3,7]
   - 0.708770215775294 W1[3,8] - 0.1431588447320848 W1[3,9]
   - 0.5543644040322324 W1[3,10] + 0.9741067643275577 W1[3,11]
   + 1.004490912277293 W1[3,12] + 1.07764214286193 W1[3,13]
   + 1.127253055081355 W1[3,14] + 0.3229769328253679 W1[3,15] - b1[3]
   + z1[3,3] = 0
 R76: - z1[3,3] + a1=max(0,z)[3,3] >= 0
 R77: a1=max(0,z)[3,3] >= 0
 R78: - z1[3,3] + a1=max(0,z)[3,3] + 16 binary_vars1[3,3] <= 16
 R79: a1=max(0,z)[3,3] - 16 binary_vars1[3,3] <= 0
 R80: - z_final[0,0] + y_pred[0,0] >= 0
 R81: y_pred[0,0] >= 0
 R82: - z_final[0,0] + y_pred[0,0] + 64 binary_vars_final[0,0] <= 64
 R83: y_pred[0,0] - 64 binary_vars_final[0,0] <= 0
 R84: - z_final[0,1] + y_pred[0,1] >= 0
 R85: y_pred[0,1] >= 0
 R86: - z_final[0,1] + y_pred[0,1] + 64 binary_vars_final[0,1] <= 64
 R87: y_pred[0,1] - 64 binary_vars_final[0,1] <= 0
 R88: - z_final[0,2] + y_pred[0,2] >= 0
 R89: y_pred[0,2] >= 0
 R90: - z_final[0,2] + y_pred[0,2] + 64 binary_vars_final[0,2] <= 64
 R91: y_pred[0,2] - 64 binary_vars_final[0,2] <= 0
 R92: - z_final[0,3] + y_pred[0,3] >= 0
 R93: y_pred[0,3] >= 0
 R94: - z_final[0,3] + y_pred[0,3] + 64 binary_vars_final[0,3] <= 64
 R95: y_pred[0,3] - 64 binary_vars_final[0,3] <= 0
 R96: - z_final[0,4] + y_pred[0,4] >= 0
 R97: y_pred[0,4] >= 0
 R98: - z_final[0,4] + y_pred[0,4] + 64 binary_vars_final[0,4] <= 64
 R99: y_pred[0,4] - 64 binary_vars_final[0,4] <= 0
 R100: - z_final[0,5] + y_pred[0,5] >= 0
 R101: y_pred[0,5] >= 0
 R102: - z_final[0,5] + y_pred[0,5] + 64 binary_vars_final[0,5] <= 64
 R103: y_pred[0,5] - 64 binary_vars_final[0,5] <= 0
 R104: - z_final[0,6] + y_pred[0,6] >= 0
 R105: y_pred[0,6] >= 0
 R106: - z_final[0,6] + y_pred[0,6] + 64 binary_vars_final[0,6] <= 64
 R107: y_pred[0,6] - 64 binary_vars_final[0,6] <= 0
 R108: - z_final[0,7] + y_pred[0,7] >= 0
 R109: y_pred[0,7] >= 0
 R110: - z_final[0,7] + y_pred[0,7] + 64 binary_vars_final[0,7] <= 64
 R111: y_pred[0,7] - 64 binary_vars_final[0,7] <= 0
 R112: - z_final[0,8] + y_pred[0,8] >= 0
 R113: y_pred[0,8] >= 0
 R114: - z_final[0,8] + y_pred[0,8] + 64 binary_vars_final[0,8] <= 64
 R115: y_pred[0,8] - 64 binary_vars_final[0,8] <= 0
 R116: - z_final[0,9] + y_pred[0,9] >= 0
 R117: y_pred[0,9] >= 0
 R118: - z_final[0,9] + y_pred[0,9] + 64 binary_vars_final[0,9] <= 64
 R119: y_pred[0,9] - 64 binary_vars_final[0,9] <= 0
 R120: - z_final[1,0] + y_pred[1,0] >= 0
 R121: y_pred[1,0] >= 0
 R122: - z_final[1,0] + y_pred[1,0] + 64 binary_vars_final[1,0] <= 64
 R123: y_pred[1,0] - 64 binary_vars_final[1,0] <= 0
 R124: - z_final[1,1] + y_pred[1,1] >= 0
 R125: y_pred[1,1] >= 0
 R126: - z_final[1,1] + y_pred[1,1] + 64 binary_vars_final[1,1] <= 64
 R127: y_pred[1,1] - 64 binary_vars_final[1,1] <= 0
 R128: - z_final[1,2] + y_pred[1,2] >= 0
 R129: y_pred[1,2] >= 0
 R130: - z_final[1,2] + y_pred[1,2] + 64 binary_vars_final[1,2] <= 64
 R131: y_pred[1,2] - 64 binary_vars_final[1,2] <= 0
 R132: - z_final[1,3] + y_pred[1,3] >= 0
 R133: y_pred[1,3] >= 0
 R134: - z_final[1,3] + y_pred[1,3] + 64 binary_vars_final[1,3] <= 64
 R135: y_pred[1,3] - 64 binary_vars_final[1,3] <= 0
 R136: - z_final[1,4] + y_pred[1,4] >= 0
 R137: y_pred[1,4] >= 0
 R138: - z_final[1,4] + y_pred[1,4] + 64 binary_vars_final[1,4] <= 64
 R139: y_pred[1,4] - 64 binary_vars_final[1,4] <= 0
 R140: - z_final[1,5] + y_pred[1,5] >= 0
 R141: y_pred[1,5] >= 0
 R142: - z_final[1,5] + y_pred[1,5] + 64 binary_vars_final[1,5] <= 64
 R143: y_pred[1,5] - 64 binary_vars_final[1,5] <= 0
 R144: - z_final[1,6] + y_pred[1,6] >= 0
 R145: y_pred[1,6] >= 0
 R146: - z_final[1,6] + y_pred[1,6] + 64 binary_vars_final[1,6] <= 64
 R147: y_pred[1,6] - 64 binary_vars_final[1,6] <= 0
 R148: - z_final[1,7] + y_pred[1,7] >= 0
 R149: y_pred[1,7] >= 0
 R150: - z_final[1,7] + y_pred[1,7] + 64 binary_vars_final[1,7] <= 64
 R151: y_pred[1,7] - 64 binary_vars_final[1,7] <= 0
 R152: - z_final[1,8] + y_pred[1,8] >= 0
 R153: y_pred[1,8] >= 0
 R154: - z_final[1,8] + y_pred[1,8] + 64 binary_vars_final[1,8] <= 64
 R155: y_pred[1,8] - 64 binary_vars_final[1,8] <= 0
 R156: - z_final[1,9] + y_pred[1,9] >= 0
 R157: y_pred[1,9] >= 0
 R158: - z_final[1,9] + y_pred[1,9] + 64 binary_vars_final[1,9] <= 64
 R159: y_pred[1,9] - 64 binary_vars_final[1,9] <= 0
 R160: - z_final[2,0] + y_pred[2,0] >= 0
 R161: y_pred[2,0] >= 0
 R162: - z_final[2,0] + y_pred[2,0] + 64 binary_vars_final[2,0] <= 64
 R163: y_pred[2,0] - 64 binary_vars_final[2,0] <= 0
 R164: - z_final[2,1] + y_pred[2,1] >= 0
 R165: y_pred[2,1] >= 0
 R166: - z_final[2,1] + y_pred[2,1] + 64 binary_vars_final[2,1] <= 64
 R167: y_pred[2,1] - 64 binary_vars_final[2,1] <= 0
 R168: - z_final[2,2] + y_pred[2,2] >= 0
 R169: y_pred[2,2] >= 0
 R170: - z_final[2,2] + y_pred[2,2] + 64 binary_vars_final[2,2] <= 64
 R171: y_pred[2,2] - 64 binary_vars_final[2,2] <= 0
 R172: - z_final[2,3] + y_pred[2,3] >= 0
 R173: y_pred[2,3] >= 0
 R174: - z_final[2,3] + y_pred[2,3] + 64 binary_vars_final[2,3] <= 64
 R175: y_pred[2,3] - 64 binary_vars_final[2,3] <= 0
 R176: - z_final[2,4] + y_pred[2,4] >= 0
 R177: y_pred[2,4] >= 0
 R178: - z_final[2,4] + y_pred[2,4] + 64 binary_vars_final[2,4] <= 64
 R179: y_pred[2,4] - 64 binary_vars_final[2,4] <= 0
 R180: - z_final[2,5] + y_pred[2,5] >= 0
 R181: y_pred[2,5] >= 0
 R182: - z_final[2,5] + y_pred[2,5] + 64 binary_vars_final[2,5] <= 64
 R183: y_pred[2,5] - 64 binary_vars_final[2,5] <= 0
 R184: - z_final[2,6] + y_pred[2,6] >= 0
 R185: y_pred[2,6] >= 0
 R186: - z_final[2,6] + y_pred[2,6] + 64 binary_vars_final[2,6] <= 64
 R187: y_pred[2,6] - 64 binary_vars_final[2,6] <= 0
 R188: - z_final[2,7] + y_pred[2,7] >= 0
 R189: y_pred[2,7] >= 0
 R190: - z_final[2,7] + y_pred[2,7] + 64 binary_vars_final[2,7] <= 64
 R191: y_pred[2,7] - 64 binary_vars_final[2,7] <= 0
 R192: - z_final[2,8] + y_pred[2,8] >= 0
 R193: y_pred[2,8] >= 0
 R194: - z_final[2,8] + y_pred[2,8] + 64 binary_vars_final[2,8] <= 64
 R195: y_pred[2,8] - 64 binary_vars_final[2,8] <= 0
 R196: - z_final[2,9] + y_pred[2,9] >= 0
 R197: y_pred[2,9] >= 0
 R198: - z_final[2,9] + y_pred[2,9] + 64 binary_vars_final[2,9] <= 64
 R199: y_pred[2,9] - 64 binary_vars_final[2,9] <= 0
 R200: - z_final[3,0] + y_pred[3,0] >= 0
 R201: y_pred[3,0] >= 0
 R202: - z_final[3,0] + y_pred[3,0] + 64 binary_vars_final[3,0] <= 64
 R203: y_pred[3,0] - 64 binary_vars_final[3,0] <= 0
 R204: - z_final[3,1] + y_pred[3,1] >= 0
 R205: y_pred[3,1] >= 0
 R206: - z_final[3,1] + y_pred[3,1] + 64 binary_vars_final[3,1] <= 64
 R207: y_pred[3,1] - 64 binary_vars_final[3,1] <= 0
 R208: - z_final[3,2] + y_pred[3,2] >= 0
 R209: y_pred[3,2] >= 0
 R210: - z_final[3,2] + y_pred[3,2] + 64 binary_vars_final[3,2] <= 64
 R211: y_pred[3,2] - 64 binary_vars_final[3,2] <= 0
 R212: - z_final[3,3] + y_pred[3,3] >= 0
 R213: y_pred[3,3] >= 0
 R214: - z_final[3,3] + y_pred[3,3] + 64 binary_vars_final[3,3] <= 64
 R215: y_pred[3,3] - 64 binary_vars_final[3,3] <= 0
 R216: - z_final[3,4] + y_pred[3,4] >= 0
 R217: y_pred[3,4] >= 0
 R218: - z_final[3,4] + y_pred[3,4] + 64 binary_vars_final[3,4] <= 64
 R219: y_pred[3,4] - 64 binary_vars_final[3,4] <= 0
 R220: - z_final[3,5] + y_pred[3,5] >= 0
 R221: y_pred[3,5] >= 0
 R222: - z_final[3,5] + y_pred[3,5] + 64 binary_vars_final[3,5] <= 64
 R223: y_pred[3,5] - 64 binary_vars_final[3,5] <= 0
 R224: - z_final[3,6] + y_pred[3,6] >= 0
 R225: y_pred[3,6] >= 0
 R226: - z_final[3,6] + y_pred[3,6] + 64 binary_vars_final[3,6] <= 64
 R227: y_pred[3,6] - 64 binary_vars_final[3,6] <= 0
 R228: - z_final[3,7] + y_pred[3,7] >= 0
 R229: y_pred[3,7] >= 0
 R230: - z_final[3,7] + y_pred[3,7] + 64 binary_vars_final[3,7] <= 64
 R231: y_pred[3,7] - 64 binary_vars_final[3,7] <= 0
 R232: - z_final[3,8] + y_pred[3,8] >= 0
 R233: y_pred[3,8] >= 0
 R234: - z_final[3,8] + y_pred[3,8] + 64 binary_vars_final[3,8] <= 64
 R235: y_pred[3,8] - 64 binary_vars_final[3,8] <= 0
 R236: - z_final[3,9] + y_pred[3,9] >= 0
 R237: y_pred[3,9] >= 0
 R238: - z_final[3,9] + y_pred[3,9] + 64 binary_vars_final[3,9] <= 64
 R239: y_pred[3,9] - 64 binary_vars_final[3,9] <= 0
 R240: hinge_loss_terms[0,0] >= 0
 R241: y_pred[0,0] + hinge_loss_terms[0,0] >= 1
 R242: hinge_loss_terms[0,1] >= 0
 R243: - y_pred[0,1] + hinge_loss_terms[0,1] >= 1
 R244: hinge_loss_terms[0,2] >= 0
 R245: - y_pred[0,2] + hinge_loss_terms[0,2] >= 1
 R246: hinge_loss_terms[0,3] >= 0
 R247: - y_pred[0,3] + hinge_loss_terms[0,3] >= 1
 R248: hinge_loss_terms[0,4] >= 0
 R249: - y_pred[0,4] + hinge_loss_terms[0,4] >= 1
 R250: hinge_loss_terms[0,5] >= 0
 R251: - y_pred[0,5] + hinge_loss_terms[0,5] >= 1
 R252: hinge_loss_terms[0,6] >= 0
 R253: - y_pred[0,6] + hinge_loss_terms[0,6] >= 1
 R254: hinge_loss_terms[0,7] >= 0
 R255: - y_pred[0,7] + hinge_loss_terms[0,7] >= 1
 R256: hinge_loss_terms[0,8] >= 0
 R257: - y_pred[0,8] + hinge_loss_terms[0,8] >= 1
 R258: hinge_loss_terms[0,9] >= 0
 R259: - y_pred[0,9] + hinge_loss_terms[0,9] >= 1
 R260: hinge_loss_terms[1,0] >= 0
 R261: - y_pred[1,0] + hinge_loss_terms[1,0] >= 1
 R262: hinge_loss_terms[1,1] >= 0
 R263: y_pred[1,1] + hinge_loss_terms[1,1] >= 1
 R264: hinge_loss_terms[1,2] >= 0
 R265: - y_pred[1,2] + hinge_loss_terms[1,2] >= 1
 R266: hinge_loss_terms[1,3] >= 0
 R267: - y_pred[1,3] + hinge_loss_terms[1,3] >= 1
 R268: hinge_loss_terms[1,4] >= 0
 R269: - y_pred[1,4] + hinge_loss_terms[1,4] >= 1
 R270: hinge_loss_terms[1,5] >= 0
 R271: - y_pred[1,5] + hinge_loss_terms[1,5] >= 1
 R272: hinge_loss_terms[1,6] >= 0
 R273: - y_pred[1,6] + hinge_loss_terms[1,6] >= 1
 R274: hinge_loss_terms[1,7] >= 0
 R275: - y_pred[1,7] + hinge_loss_terms[1,7] >= 1
 R276: hinge_loss_terms[1,8] >= 0
 R277: - y_pred[1,8] + hinge_loss_terms[1,8] >= 1
 R278: hinge_loss_terms[1,9] >= 0
 R279: - y_pred[1,9] + hinge_loss_terms[1,9] >= 1
 R280: hinge_loss_terms[2,0] >= 0
 R281: - y_pred[2,0] + hinge_loss_terms[2,0] >= 1
 R282: hinge_loss_terms[2,1] >= 0
 R283: - y_pred[2,1] + hinge_loss_terms[2,1] >= 1
 R284: hinge_loss_terms[2,2] >= 0
 R285: y_pred[2,2] + hinge_loss_terms[2,2] >= 1
 R286: hinge_loss_terms[2,3] >= 0
 R287: - y_pred[2,3] + hinge_loss_terms[2,3] >= 1
 R288: hinge_loss_terms[2,4] >= 0
 R289: - y_pred[2,4] + hinge_loss_terms[2,4] >= 1
 R290: hinge_loss_terms[2,5] >= 0
 R291: - y_pred[2,5] + hinge_loss_terms[2,5] >= 1
 R292: hinge_loss_terms[2,6] >= 0
 R293: - y_pred[2,6] + hinge_loss_terms[2,6] >= 1
 R294: hinge_loss_terms[2,7] >= 0
 R295: - y_pred[2,7] + hinge_loss_terms[2,7] >= 1
 R296: hinge_loss_terms[2,8] >= 0
 R297: - y_pred[2,8] + hinge_loss_terms[2,8] >= 1
 R298: hinge_loss_terms[2,9] >= 0
 R299: - y_pred[2,9] + hinge_loss_terms[2,9] >= 1
 R300: hinge_loss_terms[3,0] >= 0
 R301: - y_pred[3,0] + hinge_loss_terms[3,0] >= 1
 R302: hinge_loss_terms[3,1] >= 0
 R303: - y_pred[3,1] + hinge_loss_terms[3,1] >= 1
 R304: hinge_loss_terms[3,2] >= 0
 R305: - y_pred[3,2] + hinge_loss_terms[3,2] >= 1
 R306: hinge_loss_terms[3,3] >= 0
 R307: y_pred[3,3] + hinge_loss_terms[3,3] >= 1
 R308: hinge_loss_terms[3,4] >= 0
 R309: - y_pred[3,4] + hinge_loss_terms[3,4] >= 1
 R310: hinge_loss_terms[3,5] >= 0
 R311: - y_pred[3,5] + hinge_loss_terms[3,5] >= 1
 R312: hinge_loss_terms[3,6] >= 0
 R313: - y_pred[3,6] + hinge_loss_terms[3,6] >= 1
 R314: hinge_loss_terms[3,7] >= 0
 R315: - y_pred[3,7] + hinge_loss_terms[3,7] >= 1
 R316: hinge_loss_terms[3,8] >= 0
 R317: - y_pred[3,8] + hinge_loss_terms[3,8] >= 1
 R318: hinge_loss_terms[3,9] >= 0
 R319: - y_pred[3,9] + hinge_loss_terms[3,9] >= 1
 R320: - W1[0,0] + abs_W1[0,0] >= 0
 R321: W1[0,0] + abs_W1[0,0] >= 0
 R322: - W1[0,1] + abs_W1[0,1] >= 0
 R323: W1[0,1] + abs_W1[0,1] >= 0
 R324: - W1[0,2] + abs_W1[0,2] >= 0
 R325: W1[0,2] + abs_W1[0,2] >= 0
 R326: - W1[0,3] + abs_W1[0,3] >= 0
 R327: W1[0,3] + abs_W1[0,3] >= 0
 R328: - W1[0,4] + abs_W1[0,4] >= 0
 R329: W1[0,4] + abs_W1[0,4] >= 0
 R330: - W1[0,5] + abs_W1[0,5] >= 0
 R331: W1[0,5] + abs_W1[0,5] >= 0
 R332: - W1[0,6] + abs_W1[0,6] >= 0
 R333: W1[0,6] + abs_W1[0,6] >= 0
 R334: - W1[0,7] + abs_W1[0,7] >= 0
 R335: W1[0,7] + abs_W1[0,7] >= 0
 R336: - W1[0,8] + abs_W1[0,8] >= 0
 R337: W1[0,8] + abs_W1[0,8] >= 0
 R338: - W1[0,9] + abs_W1[0,9] >= 0
 R339: W1[0,9] + abs_W1[0,9] >= 0
 R340: - W1[0,10] + abs_W1[0,10] >= 0
 R341: W1[0,10] + abs_W1[0,10] >= 0
 R342: - W1[0,11] + abs_W1[0,11] >= 0
 R343: W1[0,11] + abs_W1[0,11] >= 0
 R344: - W1[0,12] + abs_W1[0,12] >= 0
 R345: W1[0,12] + abs_W1[0,12] >= 0
 R346: - W1[0,13] + abs_W1[0,13] >= 0
 R347: W1[0,13] + abs_W1[0,13] >= 0
 R348: - W1[0,14] + abs_W1[0,14] >= 0
 R349: W1[0,14] + abs_W1[0,14] >= 0
 R350: - W1[0,15] + abs_W1[0,15] >= 0
 R351: W1[0,15] + abs_W1[0,15] >= 0
 R352: - W1[1,0] + abs_W1[1,0] >= 0
 R353: W1[1,0] + abs_W1[1,0] >= 0
 R354: - W1[1,1] + abs_W1[1,1] >= 0
 R355: W1[1,1] + abs_W1[1,1] >= 0
 R356: - W1[1,2] + abs_W1[1,2] >= 0
 R357: W1[1,2] + abs_W1[1,2] >= 0
 R358: - W1[1,3] + abs_W1[1,3] >= 0
 R359: W1[1,3] + abs_W1[1,3] >= 0
 R360: - W1[1,4] + abs_W1[1,4] >= 0
 R361: W1[1,4] + abs_W1[1,4] >= 0
 R362: - W1[1,5] + abs_W1[1,5] >= 0
 R363: W1[1,5] + abs_W1[1,5] >= 0
 R364: - W1[1,6] + abs_W1[1,6] >= 0
 R365: W1[1,6] + abs_W1[1,6] >= 0
 R366: - W1[1,7] + abs_W1[1,7] >= 0
 R367: W1[1,7] + abs_W1[1,7] >= 0
 R368: - W1[1,8] + abs_W1[1,8] >= 0
 R369: W1[1,8] + abs_W1[1,8] >= 0
 R370: - W1[1,9] + abs_W1[1,9] >= 0
 R371: W1[1,9] + abs_W1[1,9] >= 0
 R372: - W1[1,10] + abs_W1[1,10] >= 0
 R373: W1[1,10] + abs_W1[1,10] >= 0
 R374: - W1[1,11] + abs_W1[1,11] >= 0
 R375: W1[1,11] + abs_W1[1,11] >= 0
 R376: - W1[1,12] + abs_W1[1,12] >= 0
 R377: W1[1,12] + abs_W1[1,12] >= 0
 R378: - W1[1,13] + abs_W1[1,13] >= 0
 R379: W1[1,13] + abs_W1[1,13] >= 0
 R380: - W1[1,14] + abs_W1[1,14] >= 0
 R381: W1[1,14] + abs_W1[1,14] >= 0
 R382: - W1[1,15] + abs_W1[1,15] >= 0
 R383: W1[1,15] + abs_W1[1,15] >= 0
 R384: - W1[2,0] + abs_W1[2,0] >= 0
 R385: W1[2,0] + abs_W1[2,0] >= 0
 R386: - W1[2,1] + abs_W1[2,1] >= 0
 R387: W1[2,1] + abs_W1[2,1] >= 0
 R388: - W1[2,2] + abs_W1[2,2] >= 0
 R389: W1[2,2] + abs_W1[2,2] >= 0
 R390: - W1[2,3] + abs_W1[2,3] >= 0
 R391: W1[2,3] + abs_W1[2,3] >= 0
 R392: - W1[2,4] + abs_W1[2,4] >= 0
 R393: W1[2,4] + abs_W1[2,4] >= 0
 R394: - W1[2,5] + abs_W1[2,5] >= 0
 R395: W1[2,5] + abs_W1[2,5] >= 0
 R396: - W1[2,6] + abs_W1[2,6] >= 0
 R397: W1[2,6] + abs_W1[2,6] >= 0
 R398: - W1[2,7] + abs_W1[2,7] >= 0
 R399: W1[2,7] + abs_W1[2,7] >= 0
 R400: - W1[2,8] + abs_W1[2,8] >= 0
 R401: W1[2,8] + abs_W1[2,8] >= 0
 R402: - W1[2,9] + abs_W1[2,9] >= 0
 R403: W1[2,9] + abs_W1[2,9] >= 0
 R404: - W1[2,10] + abs_W1[2,10] >= 0
 R405: W1[2,10] + abs_W1[2,10] >= 0
 R406: - W1[2,11] + abs_W1[2,11] >= 0
 R407: W1[2,11] + abs_W1[2,11] >= 0
 R408: - W1[2,12] + abs_W1[2,12] >= 0
 R409: W1[2,12] + abs_W1[2,12] >= 0
 R410: - W1[2,13] + abs_W1[2,13] >= 0
 R411: W1[2,13] + abs_W1[2,13] >= 0
 R412: - W1[2,14] + abs_W1[2,14] >= 0
 R413: W1[2,14] + abs_W1[2,14] >= 0
 R414: - W1[2,15] + abs_W1[2,15] >= 0
 R415: W1[2,15] + abs_W1[2,15] >= 0
 R416: - W1[3,0] + abs_W1[3,0] >= 0
 R417: W1[3,0] + abs_W1[3,0] >= 0
 R418: - W1[3,1] + abs_W1[3,1] >= 0
 R419: W1[3,1] + abs_W1[3,1] >= 0
 R420: - W1[3,2] + abs_W1[3,2] >= 0
 R421: W1[3,2] + abs_W1[3,2] >= 0
 R422: - W1[3,3] + abs_W1[3,3] >= 0
 R423: W1[3,3] + abs_W1[3,3] >= 0
 R424: - W1[3,4] + abs_W1[3,4] >= 0
 R425: W1[3,4] + abs_W1[3,4] >= 0
 R426: - W1[3,5] + abs_W1[3,5] >= 0
 R427: W1[3,5] + abs_W1[3,5] >= 0
 R428: - W1[3,6] + abs_W1[3,6] >= 0
 R429: W1[3,6] + abs_W1[3,6] >= 0
 R430: - W1[3,7] + abs_W1[3,7] >= 0
 R431: W1[3,7] + abs_W1[3,7] >= 0
 R432: - W1[3,8] + abs_W1[3,8] >= 0
 R433: W1[3,8] + abs_W1[3,8] >= 0
 R434: - W1[3,9] + abs_W1[3,9] >= 0
 R435: W1[3,9] + abs_W1[3,9] >= 0
 R436: - W1[3,10] + abs_W1[3,10] >= 0
 R437: W1[3,10] + abs_W1[3,10] >= 0
 R438: - W1[3,11] + abs_W1[3,11] >= 0
 R439: W1[3,11] + abs_W1[3,11] >= 0
 R440: - W1[3,12] + abs_W1[3,12] >= 0
 R441: W1[3,12] + abs_W1[3,12] >= 0
 R442: - W1[3,13] + abs_W1[3,13] >= 0
 R443: W1[3,13] + abs_W1[3,13] >= 0
 R444: - W1[3,14] + abs_W1[3,14] >= 0
 R445: W1[3,14] + abs_W1[3,14] >= 0
 R446: - W1[3,15] + abs_W1[3,15] >= 0
 R447: W1[3,15] + abs_W1[3,15] >= 0
 R448: - W_output[0,0] + abs_W2[0,0] >= 0
 R449: W_output[0,0] + abs_W2[0,0] >= 0
 R450: - W_output[0,1] + abs_W2[0,1] >= 0
 R451: W_output[0,1] + abs_W2[0,1] >= 0
 R452: - W_output[0,2] + abs_W2[0,2] >= 0
 R453: W_output[0,2] + abs_W2[0,2] >= 0
 R454: - W_output[0,3] + abs_W2[0,3] >= 0
 R455: W_output[0,3] + abs_W2[0,3] >= 0
 R456: - W_output[1,0] + abs_W2[1,0] >= 0
 R457: W_output[1,0] + abs_W2[1,0] >= 0
 R458: - W_output[1,1] + abs_W2[1,1] >= 0
 R459: W_output[1,1] + abs_W2[1,1] >= 0
 R460: - W_output[1,2] + abs_W2[1,2] >= 0
 R461: W_output[1,2] + abs_W2[1,2] >= 0
 R462: - W_output[1,3] + abs_W2[1,3] >= 0
 R463: W_output[1,3] + abs_W2[1,3] >= 0
 R464: - W_output[2,0] + abs_W2[2,0] >= 0
 R465: W_output[2,0] + abs_W2[2,0] >= 0
 R466: - W_output[2,1] + abs_W2[2,1] >= 0
 R467: W_output[2,1] + abs_W2[2,1] >= 0
 R468: - W_output[2,2] + abs_W2[2,2] >= 0
 R469: W_output[2,2] + abs_W2[2,2] >= 0
 R470: - W_output[2,3] + abs_W2[2,3] >= 0
 R471: W_output[2,3] + abs_W2[2,3] >= 0
 R472: - W_output[3,0] + abs_W2[3,0] >= 0
 R473: W_output[3,0] + abs_W2[3,0] >= 0
 R474: - W_output[3,1] + abs_W2[3,1] >= 0
 R475: W_output[3,1] + abs_W2[3,1] >= 0
 R476: - W_output[3,2] + abs_W2[3,2] >= 0
 R477: W_output[3,2] + abs_W2[3,2] >= 0
 R478: - W_output[3,3] + abs_W2[3,3] >= 0
 R479: W_output[3,3] + abs_W2[3,3] >= 0
 R480: - W_output[4,0] + abs_W2[4,0] >= 0
 R481: W_output[4,0] + abs_W2[4,0] >= 0
 R482: - W_output[4,1] + abs_W2[4,1] >= 0
 R483: W_output[4,1] + abs_W2[4,1] >= 0
 R484: - W_output[4,2] + abs_W2[4,2] >= 0
 R485: W_output[4,2] + abs_W2[4,2] >= 0
 R486: - W_output[4,3] + abs_W2[4,3] >= 0
 R487: W_output[4,3] + abs_W2[4,3] >= 0
 R488: - W_output[5,0] + abs_W2[5,0] >= 0
 R489: W_output[5,0] + abs_W2[5,0] >= 0
 R490: - W_output[5,1] + abs_W2[5,1] >= 0
 R491: W_output[5,1] + abs_W2[5,1] >= 0
 R492: - W_output[5,2] + abs_W2[5,2] >= 0
 R493: W_output[5,2] + abs_W2[5,2] >= 0
 R494: - W_output[5,3] + abs_W2[5,3] >= 0
 R495: W_output[5,3] + abs_W2[5,3] >= 0
 R496: - W_output[6,0] + abs_W2[6,0] >= 0
 R497: W_output[6,0] + abs_W2[6,0] >= 0
 R498: - W_output[6,1] + abs_W2[6,1] >= 0
 R499: W_output[6,1] + abs_W2[6,1] >= 0
 R500: - W_output[6,2] + abs_W2[6,2] >= 0
 R501: W_output[6,2] + abs_W2[6,2] >= 0
 R502: - W_output[6,3] + abs_W2[6,3] >= 0
 R503: W_output[6,3] + abs_W2[6,3] >= 0
 R504: - W_output[7,0] + abs_W2[7,0] >= 0
 R505: W_output[7,0] + abs_W2[7,0] >= 0
 R506: - W_output[7,1] + abs_W2[7,1] >= 0
 R507: W_output[7,1] + abs_W2[7,1] >= 0
 R508: - W_output[7,2] + abs_W2[7,2] >= 0
 R509: W_output[7,2] + abs_W2[7,2] >= 0
 R510: - W_output[7,3] + abs_W2[7,3] >= 0
 R511: W_output[7,3] + abs_W2[7,3] >= 0
 R512: - W_output[8,0] + abs_W2[8,0] >= 0
 R513: W_output[8,0] + abs_W2[8,0] >= 0
 R514: - W_output[8,1] + abs_W2[8,1] >= 0
 R515: W_output[8,1] + abs_W2[8,1] >= 0
 R516: - W_output[8,2] + abs_W2[8,2] >= 0
 R517: W_output[8,2] + abs_W2[8,2] >= 0
 R518: - W_output[8,3] + abs_W2[8,3] >= 0
 R519: W_output[8,3] + abs_W2[8,3] >= 0
 R520: - W_output[9,0] + abs_W2[9,0] >= 0
 R521: W_output[9,0] + abs_W2[9,0] >= 0
 R522: - W_output[9,1] + abs_W2[9,1] >= 0
 R523: W_output[9,1] + abs_W2[9,1] >= 0
 R524: - W_output[9,2] + abs_W2[9,2] >= 0
 R525: W_output[9,2] + abs_W2[9,2] >= 0
 R526: - W_output[9,3] + abs_W2[9,3] >= 0
 R527: W_output[9,3] + abs_W2[9,3] >= 0
 R528: - b1[0] + abs_b1[0] >= 0
 R529: b1[0] + abs_b1[0] >= 0
 R530: - b1[1] + abs_b1[1] >= 0
 R531: b1[1] + abs_b1[1] >= 0
 R532: - b1[2] + abs_b1[2] >= 0
 R533: b1[2] + abs_b1[2] >= 0
 R534: - b1[3] + abs_b1[3] >= 0
 R535: b1[3] + abs_b1[3] >= 0
 R536: - b_output[0] + abs_b2[0] >= 0
 R537: b_output[0] + abs_b2[0] >= 0
 R538: - b_output[1] + abs_b2[1] >= 0
 R539: b_output[1] + abs_b2[1] >= 0
 R540: - b_output[2] + abs_b2[2] >= 0
 R541: b_output[2] + abs_b2[2] >= 0
 R542: - b_output[3] + abs_b2[3] >= 0
 R543: b_output[3] + abs_b2[3] >= 0
 R544: - b_output[4] + abs_b2[4] >= 0
 R545: b_output[4] + abs_b2[4] >= 0
 R546: - b_output[5] + abs_b2[5] >= 0
 R547: b_output[5] + abs_b2[5] >= 0
 R548: - b_output[6] + abs_b2[6] >= 0
 R549: b_output[6] + abs_b2[6] >= 0
 R550: - b_output[7] + abs_b2[7] >= 0
 R551: b_output[7] + abs_b2[7] >= 0
 R552: - b_output[8] + abs_b2[8] >= 0
 R553: b_output[8] + abs_b2[8] >= 0
 R554: - b_output[9] + abs_b2[9] >= 0
 R555: b_output[9] + abs_b2[9] >= 0
 qc0: - b_output[0] + z_final[0,0] + [ - a1=max(0,z)[0,0] * W_output[0,0]
   - a1=max(0,z)[0,1] * W_output[0,1] - a1=max(0,z)[0,2] * W_output[0,2]
   - a1=max(0,z)[0,3] * W_output[0,3] ] = 0
 qc1: - b_output[1] + z_final[0,1] + [ - a1=max(0,z)[0,0] * W_output[1,0]
   - a1=max(0,z)[0,1] * W_output[1,1] - a1=max(0,z)[0,2] * W_output[1,2]
   - a1=max(0,z)[0,3] * W_output[1,3] ] = 0
 qc2: - b_output[2] + z_final[0,2] + [ - a1=max(0,z)[0,0] * W_output[2,0]
   - a1=max(0,z)[0,1] * W_output[2,1] - a1=max(0,z)[0,2] * W_output[2,2]
   - a1=max(0,z)[0,3] * W_output[2,3] ] = 0
 qc3: - b_output[3] + z_final[0,3] + [ - a1=max(0,z)[0,0] * W_output[3,0]
   - a1=max(0,z)[0,1] * W_output[3,1] - a1=max(0,z)[0,2] * W_output[3,2]
   - a1=max(0,z)[0,3] * W_output[3,3] ] = 0
 qc4: - b_output[4] + z_final[0,4] + [ - a1=max(0,z)[0,0] * W_output[4,0]
   - a1=max(0,z)[0,1] * W_output[4,1] - a1=max(0,z)[0,2] * W_output[4,2]
   - a1=max(0,z)[0,3] * W_output[4,3] ] = 0
 qc5: - b_output[5] + z_final[0,5] + [ - a1=max(0,z)[0,0] * W_output[5,0]
   - a1=max(0,z)[0,1] * W_output[5,1] - a1=max(0,z)[0,2] * W_output[5,2]
   - a1=max(0,z)[0,3] * W_output[5,3] ] = 0
 qc6: - b_output[6] + z_final[0,6] + [ - a1=max(0,z)[0,0] * W_output[6,0]
   - a1=max(0,z)[0,1] * W_output[6,1] - a1=max(0,z)[0,2] * W_output[6,2]
   - a1=max(0,z)[0,3] * W_output[6,3] ] = 0
 qc7: - b_output[7] + z_final[0,7] + [ - a1=max(0,z)[0,0] * W_output[7,0]
   - a1=max(0,z)[0,1] * W_output[7,1] - a1=max(0,z)[0,2] * W_output[7,2]
   - a1=max(0,z)[0,3] * W_output[7,3] ] = 0
 qc8: - b_output[8] + z_final[0,8] + [ - a1=max(0,z)[0,0] * W_output[8,0]
   - a1=max(0,z)[0,1] * W_output[8,1] - a1=max(0,z)[0,2] * W_output[8,2]
   - a1=max(0,z)[0,3] * W_output[8,3] ] = 0
 qc9: - b_output[9] + z_final[0,9] + [ - a1=max(0,z)[0,0] * W_output[9,0]
   - a1=max(0,z)[0,1] * W_output[9,1] - a1=max(0,z)[0,2] * W_output[9,2]
   - a1=max(0,z)[0,3] * W_output[9,3] ] = 0
 qc10: - b_output[0] + z_final[1,0] + [ - a1=max(0,z)[1,0] * W_output[0,0]
   - a1=max(0,z)[1,1] * W_output[0,1] - a1=max(0,z)[1,2] * W_output[0,2]
   - a1=max(0,z)[1,3] * W_output[0,3] ] = 0
 qc11: - b_output[1] + z_final[1,1] + [ - a1=max(0,z)[1,0] * W_output[1,0]
   - a1=max(0,z)[1,1] * W_output[1,1] - a1=max(0,z)[1,2] * W_output[1,2]
   - a1=max(0,z)[1,3] * W_output[1,3] ] = 0
 qc12: - b_output[2] + z_final[1,2] + [ - a1=max(0,z)[1,0] * W_output[2,0]
   - a1=max(0,z)[1,1] * W_output[2,1] - a1=max(0,z)[1,2] * W_output[2,2]
   - a1=max(0,z)[1,3] * W_output[2,3] ] = 0
 qc13: - b_output[3] + z_final[1,3] + [ - a1=max(0,z)[1,0] * W_output[3,0]
   - a1=max(0,z)[1,1] * W_output[3,1] - a1=max(0,z)[1,2] * W_output[3,2]
   - a1=max(0,z)[1,3] * W_output[3,3] ] = 0
 qc14: - b_output[4] + z_final[1,4] + [ - a1=max(0,z)[1,0] * W_output[4,0]
   - a1=max(0,z)[1,1] * W_output[4,1] - a1=max(0,z)[1,2] * W_output[4,2]
   - a1=max(0,z)[1,3] * W_output[4,3] ] = 0
 qc15: - b_output[5] + z_final[1,5] + [ - a1=max(0,z)[1,0] * W_output[5,0]
   - a1=max(0,z)[1,1] * W_output[5,1] - a1=max(0,z)[1,2] * W_output[5,2]
   - a1=max(0,z)[1,3] * W_output[5,3] ] = 0
 qc16: - b_output[6] + z_final[1,6] + [ - a1=max(0,z)[1,0] * W_output[6,0]
   - a1=max(0,z)[1,1] * W_output[6,1] - a1=max(0,z)[1,2] * W_output[6,2]
   - a1=max(0,z)[1,3] * W_output[6,3] ] = 0
 qc17: - b_output[7] + z_final[1,7] + [ - a1=max(0,z)[1,0] * W_output[7,0]
   - a1=max(0,z)[1,1] * W_output[7,1] - a1=max(0,z)[1,2] * W_output[7,2]
   - a1=max(0,z)[1,3] * W_output[7,3] ] = 0
 qc18: - b_output[8] + z_final[1,8] + [ - a1=max(0,z)[1,0] * W_output[8,0]
   - a1=max(0,z)[1,1] * W_output[8,1] - a1=max(0,z)[1,2] * W_output[8,2]
   - a1=max(0,z)[1,3] * W_output[8,3] ] = 0
 qc19: - b_output[9] + z_final[1,9] + [ - a1=max(0,z)[1,0] * W_output[9,0]
   - a1=max(0,z)[1,1] * W_output[9,1] - a1=max(0,z)[1,2] * W_output[9,2]
   - a1=max(0,z)[1,3] * W_output[9,3] ] = 0
 qc20: - b_output[0] + z_final[2,0] + [ - a1=max(0,z)[2,0] * W_output[0,0]
   - a1=max(0,z)[2,1] * W_output[0,1] - a1=max(0,z)[2,2] * W_output[0,2]
   - a1=max(0,z)[2,3] * W_output[0,3] ] = 0
 qc21: - b_output[1] + z_final[2,1] + [ - a1=max(0,z)[2,0] * W_output[1,0]
   - a1=max(0,z)[2,1] * W_output[1,1] - a1=max(0,z)[2,2] * W_output[1,2]
   - a1=max(0,z)[2,3] * W_output[1,3] ] = 0
 qc22: - b_output[2] + z_final[2,2] + [ - a1=max(0,z)[2,0] * W_output[2,0]
   - a1=max(0,z)[2,1] * W_output[2,1] - a1=max(0,z)[2,2] * W_output[2,2]
   - a1=max(0,z)[2,3] * W_output[2,3] ] = 0
 qc23: - b_output[3] + z_final[2,3] + [ - a1=max(0,z)[2,0] * W_output[3,0]
   - a1=max(0,z)[2,1] * W_output[3,1] - a1=max(0,z)[2,2] * W_output[3,2]
   - a1=max(0,z)[2,3] * W_output[3,3] ] = 0
 qc24: - b_output[4] + z_final[2,4] + [ - a1=max(0,z)[2,0] * W_output[4,0]
   - a1=max(0,z)[2,1] * W_output[4,1] - a1=max(0,z)[2,2] * W_output[4,2]
   - a1=max(0,z)[2,3] * W_output[4,3] ] = 0
 qc25: - b_output[5] + z_final[2,5] + [ - a1=max(0,z)[2,0] * W_output[5,0]
   - a1=max(0,z)[2,1] * W_output[5,1] - a1=max(0,z)[2,2] * W_output[5,2]
   - a1=max(0,z)[2,3] * W_output[5,3] ] = 0
 qc26: - b_output[6] + z_final[2,6] + [ - a1=max(0,z)[2,0] * W_output[6,0]
   - a1=max(0,z)[2,1] * W_output[6,1] - a1=max(0,z)[2,2] * W_output[6,2]
   - a1=max(0,z)[2,3] * W_output[6,3] ] = 0
 qc27: - b_output[7] + z_final[2,7] + [ - a1=max(0,z)[2,0] * W_output[7,0]
   - a1=max(0,z)[2,1] * W_output[7,1] - a1=max(0,z)[2,2] * W_output[7,2]
   - a1=max(0,z)[2,3] * W_output[7,3] ] = 0
 qc28: - b_output[8] + z_final[2,8] + [ - a1=max(0,z)[2,0] * W_output[8,0]
   - a1=max(0,z)[2,1] * W_output[8,1] - a1=max(0,z)[2,2] * W_output[8,2]
   - a1=max(0,z)[2,3] * W_output[8,3] ] = 0
 qc29: - b_output[9] + z_final[2,9] + [ - a1=max(0,z)[2,0] * W_output[9,0]
   - a1=max(0,z)[2,1] * W_output[9,1] - a1=max(0,z)[2,2] * W_output[9,2]
   - a1=max(0,z)[2,3] * W_output[9,3] ] = 0
 qc30: - b_output[0] + z_final[3,0] + [ - a1=max(0,z)[3,0] * W_output[0,0]
   - a1=max(0,z)[3,1] * W_output[0,1] - a1=max(0,z)[3,2] * W_output[0,2]
   - a1=max(0,z)[3,3] * W_output[0,3] ] = 0
 qc31: - b_output[1] + z_final[3,1] + [ - a1=max(0,z)[3,0] * W_output[1,0]
   - a1=max(0,z)[3,1] * W_output[1,1] - a1=max(0,z)[3,2] * W_output[1,2]
   - a1=max(0,z)[3,3] * W_output[1,3] ] = 0
 qc32: - b_output[2] + z_final[3,2] + [ - a1=max(0,z)[3,0] * W_output[2,0]
   - a1=max(0,z)[3,1] * W_output[2,1] - a1=max(0,z)[3,2] * W_output[2,2]
   - a1=max(0,z)[3,3] * W_output[2,3] ] = 0
 qc33: - b_output[3] + z_final[3,3] + [ - a1=max(0,z)[3,0] * W_output[3,0]
   - a1=max(0,z)[3,1] * W_output[3,1] - a1=max(0,z)[3,2] * W_output[3,2]
   - a1=max(0,z)[3,3] * W_output[3,3] ] = 0
 qc34: - b_output[4] + z_final[3,4] + [ - a1=max(0,z)[3,0] * W_output[4,0]
   - a1=max(0,z)[3,1] * W_output[4,1] - a1=max(0,z)[3,2] * W_output[4,2]
   - a1=max(0,z)[3,3] * W_output[4,3] ] = 0
 qc35: - b_output[5] + z_final[3,5] + [ - a1=max(0,z)[3,0] * W_output[5,0]
   - a1=max(0,z)[3,1] * W_output[5,1] - a1=max(0,z)[3,2] * W_output[5,2]
   - a1=max(0,z)[3,3] * W_output[5,3] ] = 0
 qc36: - b_output[6] + z_final[3,6] + [ - a1=max(0,z)[3,0] * W_output[6,0]
   - a1=max(0,z)[3,1] * W_output[6,1] - a1=max(0,z)[3,2] * W_output[6,2]
   - a1=max(0,z)[3,3] * W_output[6,3] ] = 0
 qc37: - b_output[7] + z_final[3,7] + [ - a1=max(0,z)[3,0] * W_output[7,0]
   - a1=max(0,z)[3,1] * W_output[7,1] - a1=max(0,z)[3,2] * W_output[7,2]
   - a1=max(0,z)[3,3] * W_output[7,3] ] = 0
 qc38: - b_output[8] + z_final[3,8] + [ - a1=max(0,z)[3,0] * W_output[8,0]
   - a1=max(0,z)[3,1] * W_output[8,1] - a1=max(0,z)[3,2] * W_output[8,2]
   - a1=max(0,z)[3,3] * W_output[8,3] ] = 0
 qc39: - b_output[9] + z_final[3,9] + [ - a1=max(0,z)[3,0] * W_output[9,0]
   - a1=max(0,z)[3,1] * W_output[9,1] - a1=max(0,z)[3,2] * W_output[9,2]
   - a1=max(0,z)[3,3] * W_output[9,3] ] = 0
Bounds
 -1 <= W1[0,0] <= 1
 -1 <= W1[0,1] <= 1
 -1 <= W1[0,2] <= 1
 -1 <= W1[0,3] <= 1
 -1 <= W1[0,4] <= 1
 -1 <= W1[0,5] <= 1
 -1 <= W1[0,6] <= 1
 -1 <= W1[0,7] <= 1
 -1 <= W1[0,8] <= 1
 -1 <= W1[0,9] <= 1
 -1 <= W1[0,10] <= 1
 -1 <= W1[0,11] <= 1
 -1 <= W1[0,12] <= 1
 -1 <= W1[0,13] <= 1
 -1 <= W1[0,14] <= 1
 -1 <= W1[0,15] <= 1
 -1 <= W1[1,0] <= 1
 -1 <= W1[1,1] <= 1
 -1 <= W1[1,2] <= 1
 -1 <= W1[1,3] <= 1
 -1 <= W1[1,4] <= 1
 -1 <= W1[1,5] <= 1
 -1 <= W1[1,6] <= 1
 -1 <= W1[1,7] <= 1
 -1 <= W1[1,8] <= 1
 -1 <= W1[1,9] <= 1
 -1 <= W1[1,10] <= 1
 -1 <= W1[1,11] <= 1
 -1 <= W1[1,12] <= 1
 -1 <= W1[1,13] <= 1
 -1 <= W1[1,14] <= 1
 -1 <= W1[1,15] <= 1
 -1 <= W1[2,0] <= 1
 -1 <= W1[2,1] <= 1
 -1 <= W1[2,2] <= 1
 -1 <= W1[2,3] <= 1
 -1 <= W1[2,4] <= 1
 -1 <= W1[2,5] <= 1
 -1 <= W1[2,6] <= 1
 -1 <= W1[2,7] <= 1
 -1 <= W1[2,8] <= 1
 -1 <= W1[2,9] <= 1
 -1 <= W1[2,10] <= 1
 -1 <= W1[2,11] <= 1
 -1 <= W1[2,12] <= 1
 -1 <= W1[2,13] <= 1
 -1 <= W1[2,14] <= 1
 -1 <= W1[2,15] <= 1
 -1 <= W1[3,0] <= 1
 -1 <= W1[3,1] <= 1
 -1 <= W1[3,2] <= 1
 -1 <= W1[3,3] <= 1
 -1 <= W1[3,4] <= 1
 -1 <= W1[3,5] <= 1
 -1 <= W1[3,6] <= 1
 -1 <= W1[3,7] <= 1
 -1 <= W1[3,8] <= 1
 -1 <= W1[3,9] <= 1
 -1 <= W1[3,10] <= 1
 -1 <= W1[3,11] <= 1
 -1 <= W1[3,12] <= 1
 -1 <= W1[3,13] <= 1
 -1 <= W1[3,14] <= 1
 -1 <= W1[3,15] <= 1
 -1 <= b1[0] <= 1
 -1 <= b1[1] <= 1
 -1 <= b1[2] <= 1
 -1 <= b1[3] <= 1
 z1[0,0] free
 z1[0,1] free
 z1[0,2] free
 z1[0,3] free
 z1[1,0] free
 z1[1,1] free
 z1[1,2] free
 z1[1,3] free
 z1[2,0] free
 z1[2,1] free
 z1[2,2] free
 z1[2,3] free
 z1[3,0] free
 z1[3,1] free
 z1[3,2] free
 z1[3,3] free
 -1 <= W_output[0,0] <= 1
 -1 <= W_output[0,1] <= 1
 -1 <= W_output[0,2] <= 1
 -1 <= W_output[0,3] <= 1
 -1 <= W_output[1,0] <= 1
 -1 <= W_output[1,1] <= 1
 -1 <= W_output[1,2] <= 1
 -1 <= W_output[1,3] <= 1
 -1 <= W_output[2,0] <= 1
 -1 <= W_output[2,1] <= 1
 -1 <= W_output[2,2] <= 1
 -1 <= W_output[2,3] <= 1
 -1 <= W_output[3,0] <= 1
 -1 <= W_output[3,1] <= 1
 -1 <= W_output[3,2] <= 1
 -1 <= W_output[3,3] <= 1
 -1 <= W_output[4,0] <= 1
 -1 <= W_output[4,1] <= 1
 -1 <= W_output[4,2] <= 1
 -1 <= W_output[4,3] <= 1
 -1 <= W_output[5,0] <= 1
 -1 <= W_output[5,1] <= 1
 -1 <= W_output[5,2] <= 1
 -1 <= W_output[5,3] <= 1
 -1 <= W_output[6,0] <= 1
 -1 <= W_output[6,1] <= 1
 -1 <= W_output[6,2] <= 1
 -1 <= W_output[6,3] <= 1
 -1 <= W_output[7,0] <= 1
 -1 <= W_output[7,1] <= 1
 -1 <= W_output[7,2] <= 1
 -1 <= W_output[7,3] <= 1
 -1 <= W_output[8,0] <= 1
 -1 <= W_output[8,1] <= 1
 -1 <= W_output[8,2] <= 1
 -1 <= W_output[8,3] <= 1
 -1 <= W_output[9,0] <= 1
 -1 <= W_output[9,1] <= 1
 -1 <= W_output[9,2] <= 1
 -1 <= W_output[9,3] <= 1
 -1 <= b_output[0] <= 1
 -1 <= b_output[1] <= 1
 -1 <= b_output[2] <= 1
 -1 <= b_output[3] <= 1
 -1 <= b_output[4] <= 1
 -1 <= b_output[5] <= 1
 -1 <= b_output[6] <= 1
 -1 <= b_output[7] <= 1
 -1 <= b_output[8] <= 1
 -1 <= b_output[9] <= 1
 z_final[0,0] free
 z_final[0,1] free
 z_final[0,2] free
 z_final[0,3] free
 z_final[0,4] free
 z_final[0,5] free
 z_final[0,6] free
 z_final[0,7] free
 z_final[0,8] free
 z_final[0,9] free
 z_final[1,0] free
 z_final[1,1] free
 z_final[1,2] free
 z_final[1,3] free
 z_final[1,4] free
 z_final[1,5] free
 z_final[1,6] free
 z_final[1,7] free
 z_final[1,8] free
 z_final[1,9] free
 z_final[2,0] free
 z_final[2,1] free
 z_final[2,2] free
 z_final[2,3] free
 z_final[2,4] free
 z_final[2,5] free
 z_final[2,6] free
 z_final[2,7] free
 z_final[2,8] free
 z_final[2,9] free
 z_final[3,0] free
 z_final[3,1] free
 z_final[3,2] free
 z_final[3,3] free
 z_final[3,4] free
 z_final[3,5] free
 z_final[3,6] free
 z_final[3,7] free
 z_final[3,8] free
 z_final[3,9] free
 abs_W1[0,0] <= 1
 abs_W1[0,1] <= 1
 abs_W1[0,2] <= 1
 abs_W1[0,3] <= 1
 abs_W1[0,4] <= 1
 abs_W1[0,5] <= 1
 abs_W1[0,6] <= 1
 abs_W1[0,7] <= 1
 abs_W1[0,8] <= 1
 abs_W1[0,9] <= 1
 abs_W1[0,10] <= 1
 abs_W1[0,11] <= 1
 abs_W1[0,12] <= 1
 abs_W1[0,13] <= 1
 abs_W1[0,14] <= 1
 abs_W1[0,15] <= 1
 abs_W1[1,0] <= 1
 abs_W1[1,1] <= 1
 abs_W1[1,2] <= 1
 abs_W1[1,3] <= 1
 abs_W1[1,4] <= 1
 abs_W1[1,5] <= 1
 abs_W1[1,6] <= 1
 abs_W1[1,7] <= 1
 abs_W1[1,8] <= 1
 abs_W1[1,9] <= 1
 abs_W1[1,10] <= 1
 abs_W1[1,11] <= 1
 abs_W1[1,12] <= 1
 abs_W1[1,13] <= 1
 abs_W1[1,14] <= 1
 abs_W1[1,15] <= 1
 abs_W1[2,0] <= 1
 abs_W1[2,1] <= 1
 abs_W1[2,2] <= 1
 abs_W1[2,3] <= 1
 abs_W1[2,4] <= 1
 abs_W1[2,5] <= 1
 abs_W1[2,6] <= 1
 abs_W1[2,7] <= 1
 abs_W1[2,8] <= 1
 abs_W1[2,9] <= 1
 abs_W1[2,10] <= 1
 abs_W1[2,11] <= 1
 abs_W1[2,12] <= 1
 abs_W1[2,13] <= 1
 abs_W1[2,14] <= 1
 abs_W1[2,15] <= 1
 abs_W1[3,0] <= 1
 abs_W1[3,1] <= 1
 abs_W1[3,2] <= 1
 abs_W1[3,3] <= 1
 abs_W1[3,4] <= 1
 abs_W1[3,5] <= 1
 abs_W1[3,6] <= 1
 abs_W1[3,7] <= 1
 abs_W1[3,8] <= 1
 abs_W1[3,9] <= 1
 abs_W1[3,10] <= 1
 abs_W1[3,11] <= 1
 abs_W1[3,12] <= 1
 abs_W1[3,13] <= 1
 abs_W1[3,14] <= 1
 abs_W1[3,15] <= 1
 abs_b1[0] <= 1
 abs_b1[1] <= 1
 abs_b1[2] <= 1
 abs_b1[3] <= 1
 abs_W2[0,0] <= 1
 abs_W2[0,1] <= 1
 abs_W2[0,2] <= 1
 abs_W2[0,3] <= 1
 abs_W2[1,0] <= 1
 abs_W2[1,1] <= 1
 abs_W2[1,2] <= 1
 abs_W2[1,3] <= 1
 abs_W2[2,0] <= 1
 abs_W2[2,1] <= 1
 abs_W2[2,2] <= 1
 abs_W2[2,3] <= 1
 abs_W2[3,0] <= 1
 abs_W2[3,1] <= 1
 abs_W2[3,2] <= 1
 abs_W2[3,3] <= 1
 abs_W2[4,0] <= 1
 abs_W2[4,1] <= 1
 abs_W2[4,2] <= 1
 abs_W2[4,3] <= 1
 abs_W2[5,0] <= 1
 abs_W2[5,1] <= 1
 abs_W2[5,2] <= 1
 abs_W2[5,3] <= 1
 abs_W2[6,0] <= 1
 abs_W2[6,1] <= 1
 abs_W2[6,2] <= 1
 abs_W2[6,3] <= 1
 abs_W2[7,0] <= 1
 abs_W2[7,1] <= 1
 abs_W2[7,2] <= 1
 abs_W2[7,3] <= 1
 abs_W2[8,0] <= 1
 abs_W2[8,1] <= 1
 abs_W2[8,2] <= 1
 abs_W2[8,3] <= 1
 abs_W2[9,0] <= 1
 abs_W2[9,1] <= 1
 abs_W2[9,2] <= 1
 abs_W2[9,3] <= 1
 abs_b2[0] <= 1
 abs_b2[1] <= 1
 abs_b2[2] <= 1
 abs_b2[3] <= 1
 abs_b2[4] <= 1
 abs_b2[5] <= 1
 abs_b2[6] <= 1
 abs_b2[7] <= 1
 abs_b2[8] <= 1
 abs_b2[9] <= 1
Binaries
 binary_vars1[0,0] binary_vars1[0,1] binary_vars1[0,2] binary_vars1[0,3]
 binary_vars1[1,0] binary_vars1[1,1] binary_vars1[1,2] binary_vars1[1,3]
 binary_vars1[2,0] binary_vars1[2,1] binary_vars1[2,2] binary_vars1[2,3]
 binary_vars1[3,0] binary_vars1[3,1] binary_vars1[3,2] binary_vars1[3,3]
 binary_vars_final[0,0] binary_vars_final[0,1] binary_vars_final[0,2]
 binary_vars_final[0,3] binary_vars_final[0,4] binary_vars_final[0,5]
 binary_vars_final[0,6] binary_vars_final[0,7] binary_vars_final[0,8]
 binary_vars_final[0,9] binary_vars_final[1,0] binary_vars_final[1,1]
 binary_vars_final[1,2] binary_vars_final[1,3] binary_vars_final[1,4]
 binary_vars_final[1,5] binary_vars_final[1,6] binary_vars_final[1,7]
 binary_vars_final[1,8] binary_vars_final[1,9] binary_vars_final[2,0]
 binary_vars_final[2,1] binary_vars_final[2,2] binary_vars_final[2,3]
 binary_vars_final[2,4] binary_vars_final[2,5] binary_vars_final[2,6]
 binary_vars_final[2,7] binary_vars_final[2,8] binary_vars_final[2,9]
 binary_vars_final[3,0] binary_vars_final[3,1] binary_vars_final[3,2]
 binary_vars_final[3,3] binary_vars_final[3,4] binary_vars_final[3,5]
 binary_vars_final[3,6] binary_vars_final[3,7] binary_vars_final[3,8]
 binary_vars_final[3,9]
End
