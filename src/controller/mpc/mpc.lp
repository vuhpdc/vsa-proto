\* Problem: mpc *\

Maximize
 QoA: + obj_accuracy

Subject To
 one(1): + control_model(1,1) + control_model(1,2) + control_model(1,3)
 + control_model(1,4) + control_model(1,5) + control_model(1,6)
 + control_model(1,7) + control_model(1,8) + control_model(1,9)
 + control_model(1,10) + control_model(1,11) + control_model(1,12)
 + control_model(1,13) + control_model(1,14) + control_model(1,15)
 + control_model(1,16) + control_model(1,17) + control_model(1,18)
 + control_model(1,19) = 1
 one(2): + control_model(2,1) + control_model(2,2) + control_model(2,3)
 + control_model(2,4) + control_model(2,5) + control_model(2,6)
 + control_model(2,7) + control_model(2,8) + control_model(2,9)
 + control_model(2,10) + control_model(2,11) + control_model(2,12)
 + control_model(2,13) + control_model(2,14) + control_model(2,15)
 + control_model(2,16) + control_model(2,17) + control_model(2,18)
 + control_model(2,19) = 1
 one(3): + control_model(3,1) + control_model(3,2) + control_model(3,3)
 + control_model(3,4) + control_model(3,5) + control_model(3,6)
 + control_model(3,7) + control_model(3,8) + control_model(3,9)
 + control_model(3,10) + control_model(3,11) + control_model(3,12)
 + control_model(3,13) + control_model(3,14) + control_model(3,15)
 + control_model(3,16) + control_model(3,17) + control_model(3,18)
 + control_model(3,19) = 1
 one(4): + control_model(4,1) + control_model(4,2) + control_model(4,3)
 + control_model(4,4) + control_model(4,5) + control_model(4,6)
 + control_model(4,7) + control_model(4,8) + control_model(4,9)
 + control_model(4,10) + control_model(4,11) + control_model(4,12)
 + control_model(4,13) + control_model(4,14) + control_model(4,15)
 + control_model(4,16) + control_model(4,17) + control_model(4,18)
 + control_model(4,19) = 1
 one(5): + control_model(5,1) + control_model(5,2) + control_model(5,3)
 + control_model(5,4) + control_model(5,5) + control_model(5,6)
 + control_model(5,7) + control_model(5,8) + control_model(5,9)
 + control_model(5,10) + control_model(5,11) + control_model(5,12)
 + control_model(5,13) + control_model(5,14) + control_model(5,15)
 + control_model(5,16) + control_model(5,17) + control_model(5,18)
 + control_model(5,19) = 1
 def_control_model_accuracy(1): - 3.6 control_model(1,1)
 - 8.5 control_model(1,2) - 13.9 control_model(1,3)
 - 18.4 control_model(1,4) - 22.6 control_model(1,5)
 - 26.4 control_model(1,6) - 29.6 control_model(1,7)
 - 32.8 control_model(1,8) - 35.6 control_model(1,9)
 - 37.8 control_model(1,10) - 39.8 control_model(1,11)
 - 41.6 control_model(1,12) - 42.3 control_model(1,13)
 - 42.9 control_model(1,14) - 43.2 control_model(1,15)
 - 42.9 control_model(1,16) - 42.2 control_model(1,17)
 - 41.7 control_model(1,18) - 40.3 control_model(1,19)
 + control_model_accuracy(1) = -0
 def_control_model_accuracy(2): - 3.6 control_model(2,1)
 - 8.5 control_model(2,2) - 13.9 control_model(2,3)
 - 18.4 control_model(2,4) - 22.6 control_model(2,5)
 - 26.4 control_model(2,6) - 29.6 control_model(2,7)
 - 32.8 control_model(2,8) - 35.6 control_model(2,9)
 - 37.8 control_model(2,10) - 39.8 control_model(2,11)
 - 41.6 control_model(2,12) - 42.3 control_model(2,13)
 - 42.9 control_model(2,14) - 43.2 control_model(2,15)
 - 42.9 control_model(2,16) - 42.2 control_model(2,17)
 - 41.7 control_model(2,18) - 40.3 control_model(2,19)
 + control_model_accuracy(2) = -0
 def_control_model_accuracy(3): - 3.6 control_model(3,1)
 - 8.5 control_model(3,2) - 13.9 control_model(3,3)
 - 18.4 control_model(3,4) - 22.6 control_model(3,5)
 - 26.4 control_model(3,6) - 29.6 control_model(3,7)
 - 32.8 control_model(3,8) - 35.6 control_model(3,9)
 - 37.8 control_model(3,10) - 39.8 control_model(3,11)
 - 41.6 control_model(3,12) - 42.3 control_model(3,13)
 - 42.9 control_model(3,14) - 43.2 control_model(3,15)
 - 42.9 control_model(3,16) - 42.2 control_model(3,17)
 - 41.7 control_model(3,18) - 40.3 control_model(3,19)
 + control_model_accuracy(3) = -0
 def_control_model_accuracy(4): - 3.6 control_model(4,1)
 - 8.5 control_model(4,2) - 13.9 control_model(4,3)
 - 18.4 control_model(4,4) - 22.6 control_model(4,5)
 - 26.4 control_model(4,6) - 29.6 control_model(4,7)
 - 32.8 control_model(4,8) - 35.6 control_model(4,9)
 - 37.8 control_model(4,10) - 39.8 control_model(4,11)
 - 41.6 control_model(4,12) - 42.3 control_model(4,13)
 - 42.9 control_model(4,14) - 43.2 control_model(4,15)
 - 42.9 control_model(4,16) - 42.2 control_model(4,17)
 - 41.7 control_model(4,18) - 40.3 control_model(4,19)
 + control_model_accuracy(4) = -0
 def_control_model_accuracy(5): - 3.6 control_model(5,1)
 - 8.5 control_model(5,2) - 13.9 control_model(5,3)
 - 18.4 control_model(5,4) - 22.6 control_model(5,5)
 - 26.4 control_model(5,6) - 29.6 control_model(5,7)
 - 32.8 control_model(5,8) - 35.6 control_model(5,9)
 - 37.8 control_model(5,10) - 39.8 control_model(5,11)
 - 41.6 control_model(5,12) - 42.3 control_model(5,13)
 - 42.9 control_model(5,14) - 43.2 control_model(5,15)
 - 42.9 control_model(5,16) - 42.2 control_model(5,17)
 - 41.7 control_model(5,18) - 40.3 control_model(5,19)
 + control_model_accuracy(5) = -0
 def_last_model_accuracy: + last_model_accuracy = 3.6
 last_abs_smoothness_1: - control_model_accuracy(1) - abs_smoothness(1)
 + last_model_accuracy <= -0
 last_abs_smoothness_2: + control_model_accuracy(1) - abs_smoothness(1)
 - last_model_accuracy <= -0
 remain_abs_smoothness_1(2): + control_model_accuracy(1)
 - control_model_accuracy(2) - abs_smoothness(2) <= -0
 remain_abs_smoothness_1(3): + control_model_accuracy(2)
 - control_model_accuracy(3) - abs_smoothness(3) <= -0
 remain_abs_smoothness_1(4): + control_model_accuracy(3)
 - control_model_accuracy(4) - abs_smoothness(4) <= -0
 remain_abs_smoothness_1(5): + control_model_accuracy(4)
 - control_model_accuracy(5) - abs_smoothness(5) <= -0
 remain_abs_smoothness_2(2): - control_model_accuracy(1)
 + control_model_accuracy(2) - abs_smoothness(2) <= -0
 remain_abs_smoothness_2(3): - control_model_accuracy(2)
 + control_model_accuracy(3) - abs_smoothness(3) <= -0
 remain_abs_smoothness_2(4): - control_model_accuracy(3)
 + control_model_accuracy(4) - abs_smoothness(4) <= -0
 remain_abs_smoothness_2(5): - control_model_accuracy(4)
 + control_model_accuracy(5) - abs_smoothness(5) <= -0
 def_selected_model_time(1): - 14.92 control_model(1,1)
 - 14.76 control_model(1,2) - 15.2894611247756 control_model(1,3)
 - 16.2406281757502 control_model(1,4)
 - 16.1843694866787 control_model(1,5)
 - 17.5223731435 control_model(1,6)
 - 18.1578406086934 control_model(1,7)
 - 19.6488121060477 control_model(1,8)
 - 20.9051011793623 control_model(1,9)
 - 23.1587033771253 control_model(1,10)
 - 24.5000779058123 control_model(1,11)
 - 27.2395506001015 control_model(1,12)
 - 29.1739661032566 control_model(1,13)
 - 32.4150012837923 control_model(1,14)
 - 34.3138482769146 control_model(1,15)
 - 41.7229084010951 control_model(1,16)
 - 46.08132948668 control_model(1,17)
 - 63.4689839925068 control_model(1,18)
 - 5403.798193362 control_model(1,19) + selected_model_time(1) = -0
 def_selected_model_time(2): - 14.92 control_model(2,1)
 - 14.76 control_model(2,2) - 15.2894611247756 control_model(2,3)
 - 16.2406281757502 control_model(2,4)
 - 16.1843694866787 control_model(2,5)
 - 17.5223731435 control_model(2,6)
 - 18.1578406086934 control_model(2,7)
 - 19.6488121060477 control_model(2,8)
 - 20.9051011793623 control_model(2,9)
 - 23.1587033771253 control_model(2,10)
 - 24.5000779058123 control_model(2,11)
 - 27.2395506001015 control_model(2,12)
 - 29.1739661032566 control_model(2,13)
 - 32.4150012837923 control_model(2,14)
 - 34.3138482769146 control_model(2,15)
 - 41.7229084010951 control_model(2,16)
 - 46.08132948668 control_model(2,17)
 - 63.4689839925068 control_model(2,18)
 - 5403.798193362 control_model(2,19) + selected_model_time(2) = -0
 def_selected_model_time(3): - 14.92 control_model(3,1)
 - 14.76 control_model(3,2) - 15.2894611247756 control_model(3,3)
 - 16.2406281757502 control_model(3,4)
 - 16.1843694866787 control_model(3,5)
 - 17.5223731435 control_model(3,6)
 - 18.1578406086934 control_model(3,7)
 - 19.6488121060477 control_model(3,8)
 - 20.9051011793623 control_model(3,9)
 - 23.1587033771253 control_model(3,10)
 - 24.5000779058123 control_model(3,11)
 - 27.2395506001015 control_model(3,12)
 - 29.1739661032566 control_model(3,13)
 - 32.4150012837923 control_model(3,14)
 - 34.3138482769146 control_model(3,15)
 - 41.7229084010951 control_model(3,16)
 - 46.08132948668 control_model(3,17)
 - 63.4689839925068 control_model(3,18)
 - 5403.798193362 control_model(3,19) + selected_model_time(3) = -0
 def_selected_model_time(4): - 14.92 control_model(4,1)
 - 14.76 control_model(4,2) - 15.2894611247756 control_model(4,3)
 - 16.2406281757502 control_model(4,4)
 - 16.1843694866787 control_model(4,5)
 - 17.5223731435 control_model(4,6)
 - 18.1578406086934 control_model(4,7)
 - 19.6488121060477 control_model(4,8)
 - 20.9051011793623 control_model(4,9)
 - 23.1587033771253 control_model(4,10)
 - 24.5000779058123 control_model(4,11)
 - 27.2395506001015 control_model(4,12)
 - 29.1739661032566 control_model(4,13)
 - 32.4150012837923 control_model(4,14)
 - 34.3138482769146 control_model(4,15)
 - 41.7229084010951 control_model(4,16)
 - 46.08132948668 control_model(4,17)
 - 63.4689839925068 control_model(4,18)
 - 5403.798193362 control_model(4,19) + selected_model_time(4) = -0
 def_selected_model_time(5): - 14.92 control_model(5,1)
 - 14.76 control_model(5,2) - 15.2894611247756 control_model(5,3)
 - 16.2406281757502 control_model(5,4)
 - 16.1843694866787 control_model(5,5)
 - 17.5223731435 control_model(5,6)
 - 18.1578406086934 control_model(5,7)
 - 19.6488121060477 control_model(5,8)
 - 20.9051011793623 control_model(5,9)
 - 23.1587033771253 control_model(5,10)
 - 24.5000779058123 control_model(5,11)
 - 27.2395506001015 control_model(5,12)
 - 29.1739661032566 control_model(5,13)
 - 32.4150012837923 control_model(5,14)
 - 34.3138482769146 control_model(5,15)
 - 41.7229084010951 control_model(5,16)
 - 46.08132948668 control_model(5,17)
 - 63.4689839925068 control_model(5,18)
 - 5403.798193362 control_model(5,19) + selected_model_time(5) = -0
 def_selected_network_time(1): - 0.66 control_model(1,1)
 - 1.06 control_model(1,2) - 1.57 control_model(1,3)
 - 2.19 control_model(1,4) - 2.9 control_model(1,5)
 - 3.71 control_model(1,6) - 4.59 control_model(1,7)
 - 5.57 control_model(1,8) - 6.61 control_model(1,9)
 - 7.76 control_model(1,10) - 8.95 control_model(1,11)
 - 10.25 control_model(1,12) - 11.6 control_model(1,13)
 - 13.22 control_model(1,14) - 14.42 control_model(1,15)
 - 16.08 control_model(1,16) - 17.7 control_model(1,17)
 - 19.39 control_model(1,18) - 21.13 control_model(1,19)
 + selected_network_time(1) = -0
 def_selected_network_time(2): - 0.66 control_model(2,1)
 - 1.06 control_model(2,2) - 1.57 control_model(2,3)
 - 2.19 control_model(2,4) - 2.9 control_model(2,5)
 - 3.71 control_model(2,6) - 4.59 control_model(2,7)
 - 5.57 control_model(2,8) - 6.61 control_model(2,9)
 - 7.76 control_model(2,10) - 8.95 control_model(2,11)
 - 10.25 control_model(2,12) - 11.6 control_model(2,13)
 - 13.22 control_model(2,14) - 14.42 control_model(2,15)
 - 16.08 control_model(2,16) - 17.7 control_model(2,17)
 - 19.39 control_model(2,18) - 21.13 control_model(2,19)
 + selected_network_time(2) = -0
 def_selected_network_time(3): - 0.66 control_model(3,1)
 - 1.06 control_model(3,2) - 1.57 control_model(3,3)
 - 2.19 control_model(3,4) - 2.9 control_model(3,5)
 - 3.71 control_model(3,6) - 4.59 control_model(3,7)
 - 5.57 control_model(3,8) - 6.61 control_model(3,9)
 - 7.76 control_model(3,10) - 8.95 control_model(3,11)
 - 10.25 control_model(3,12) - 11.6 control_model(3,13)
 - 13.22 control_model(3,14) - 14.42 control_model(3,15)
 - 16.08 control_model(3,16) - 17.7 control_model(3,17)
 - 19.39 control_model(3,18) - 21.13 control_model(3,19)
 + selected_network_time(3) = -0
 def_selected_network_time(4): - 0.66 control_model(4,1)
 - 1.06 control_model(4,2) - 1.57 control_model(4,3)
 - 2.19 control_model(4,4) - 2.9 control_model(4,5)
 - 3.71 control_model(4,6) - 4.59 control_model(4,7)
 - 5.57 control_model(4,8) - 6.61 control_model(4,9)
 - 7.76 control_model(4,10) - 8.95 control_model(4,11)
 - 10.25 control_model(4,12) - 11.6 control_model(4,13)
 - 13.22 control_model(4,14) - 14.42 control_model(4,15)
 - 16.08 control_model(4,16) - 17.7 control_model(4,17)
 - 19.39 control_model(4,18) - 21.13 control_model(4,19)
 + selected_network_time(4) = -0
 def_selected_network_time(5): - 0.66 control_model(5,1)
 - 1.06 control_model(5,2) - 1.57 control_model(5,3)
 - 2.19 control_model(5,4) - 2.9 control_model(5,5)
 - 3.71 control_model(5,6) - 4.59 control_model(5,7)
 - 5.57 control_model(5,8) - 6.61 control_model(5,9)
 - 7.76 control_model(5,10) - 8.95 control_model(5,11)
 - 10.25 control_model(5,12) - 11.6 control_model(5,13)
 - 13.22 control_model(5,14) - 14.42 control_model(5,15)
 - 16.08 control_model(5,16) - 17.7 control_model(5,17)
 - 19.39 control_model(5,18) - 21.13 control_model(5,19)
 + selected_network_time(5) = -0
 def_last_endToEnd_time: + last_endToEnd_time = 50
 first_server_queue_wait_1: + selected_network_time(1)
 - last_endToEnd_time - 150 s_binary(1) <= -33
 first_server_queue_wait_2: - selected_network_time(1)
 + last_endToEnd_time + 150 s_binary(1) <= 183
 first_server_queue_wait_3: + selected_network_time(1)
 - last_endToEnd_time + server_queue_wait(1) - 150 s_binary(1) <= -33
 first_server_queue_wait_4: + server_queue_wait(1) + 150 s_binary(1)
 <= 150
 first_server_queue_wait: + selected_network_time(1)
 - last_endToEnd_time + server_queue_wait(1) >= -33
 def_server_queue_wait_1(2): + selected_network_time(2)
 - endToEnd_time(1) - 150 s_binary(2) <= -33
 def_server_queue_wait_1(3): + selected_network_time(3)
 - endToEnd_time(2) - 150 s_binary(3) <= -33
 def_server_queue_wait_1(4): + selected_network_time(4)
 - endToEnd_time(3) - 150 s_binary(4) <= -33
 def_server_queue_wait_1(5): + selected_network_time(5)
 - endToEnd_time(4) - 150 s_binary(5) <= -33
 def_server_queue_wait_2(2): - selected_network_time(2)
 + endToEnd_time(1) + 150 s_binary(2) <= 183
 def_server_queue_wait_2(3): - selected_network_time(3)
 + endToEnd_time(2) + 150 s_binary(3) <= 183
 def_server_queue_wait_2(4): - selected_network_time(4)
 + endToEnd_time(3) + 150 s_binary(4) <= 183
 def_server_queue_wait_2(5): - selected_network_time(5)
 + endToEnd_time(4) + 150 s_binary(5) <= 183
 def_server_queue_wait_3(2): + selected_network_time(2)
 - endToEnd_time(1) + server_queue_wait(2) - 150 s_binary(2) <= -33
 def_server_queue_wait_3(3): + selected_network_time(3)
 - endToEnd_time(2) + server_queue_wait(3) - 150 s_binary(3) <= -33
 def_server_queue_wait_3(4): + selected_network_time(4)
 - endToEnd_time(3) + server_queue_wait(4) - 150 s_binary(4) <= -33
 def_server_queue_wait_3(5): + selected_network_time(5)
 - endToEnd_time(4) + server_queue_wait(5) - 150 s_binary(5) <= -33
 def_server_queue_wait_4(2): + server_queue_wait(2) + 150 s_binary(2)
 <= 150
 def_server_queue_wait_4(3): + server_queue_wait(3) + 150 s_binary(3)
 <= 150
 def_server_queue_wait_4(4): + server_queue_wait(4) + 150 s_binary(4)
 <= 150
 def_server_queue_wait_4(5): + server_queue_wait(5) + 150 s_binary(5)
 <= 150
 def_server_queue_wait(2): + selected_network_time(2) - endToEnd_time(1)
 + server_queue_wait(2) >= -33
 def_server_queue_wait(3): + selected_network_time(3) - endToEnd_time(2)
 + server_queue_wait(3) >= -33
 def_server_queue_wait(4): + selected_network_time(4) - endToEnd_time(3)
 + server_queue_wait(4) >= -33
 def_server_queue_wait(5): + selected_network_time(5) - endToEnd_time(4)
 + server_queue_wait(5) >= -33
 def_endToEnd_time(1): - selected_model_time(1)
 - selected_network_time(1) + endToEnd_time(1) - server_queue_wait(1)
 = -0
 def_endToEnd_time(2): - selected_model_time(2)
 - selected_network_time(2) + endToEnd_time(2) - server_queue_wait(2)
 = -0
 def_endToEnd_time(3): - selected_model_time(3)
 - selected_network_time(3) + endToEnd_time(3) - server_queue_wait(3)
 = -0
 def_endToEnd_time(4): - selected_model_time(4)
 - selected_network_time(4) + endToEnd_time(4) - server_queue_wait(4)
 = -0
 def_endToEnd_time(5): - selected_model_time(5)
 - selected_network_time(5) + endToEnd_time(5) - server_queue_wait(5)
 = -0
 def_obj_accuracy: - control_model_accuracy(1)
 - control_model_accuracy(2) - control_model_accuracy(3)
 - control_model_accuracy(4) - control_model_accuracy(5) + obj_accuracy
 = -0

Bounds
 0 <= control_model(1,1) <= 1
 0 <= control_model(1,2) <= 1
 0 <= control_model(1,3) <= 1
 0 <= control_model(1,4) <= 1
 0 <= control_model(1,5) <= 1
 0 <= control_model(1,6) <= 1
 0 <= control_model(1,7) <= 1
 0 <= control_model(1,8) <= 1
 0 <= control_model(1,9) <= 1
 0 <= control_model(1,10) <= 1
 0 <= control_model(1,11) <= 1
 0 <= control_model(1,12) <= 1
 0 <= control_model(1,13) <= 1
 0 <= control_model(1,14) <= 1
 0 <= control_model(1,15) <= 1
 0 <= control_model(1,16) <= 1
 0 <= control_model(1,17) <= 1
 0 <= control_model(1,18) <= 1
 0 <= control_model(1,19) <= 1
 0 <= control_model(2,1) <= 1
 0 <= control_model(2,2) <= 1
 0 <= control_model(2,3) <= 1
 0 <= control_model(2,4) <= 1
 0 <= control_model(2,5) <= 1
 0 <= control_model(2,6) <= 1
 0 <= control_model(2,7) <= 1
 0 <= control_model(2,8) <= 1
 0 <= control_model(2,9) <= 1
 0 <= control_model(2,10) <= 1
 0 <= control_model(2,11) <= 1
 0 <= control_model(2,12) <= 1
 0 <= control_model(2,13) <= 1
 0 <= control_model(2,14) <= 1
 0 <= control_model(2,15) <= 1
 0 <= control_model(2,16) <= 1
 0 <= control_model(2,17) <= 1
 0 <= control_model(2,18) <= 1
 0 <= control_model(2,19) <= 1
 0 <= control_model(3,1) <= 1
 0 <= control_model(3,2) <= 1
 0 <= control_model(3,3) <= 1
 0 <= control_model(3,4) <= 1
 0 <= control_model(3,5) <= 1
 0 <= control_model(3,6) <= 1
 0 <= control_model(3,7) <= 1
 0 <= control_model(3,8) <= 1
 0 <= control_model(3,9) <= 1
 0 <= control_model(3,10) <= 1
 0 <= control_model(3,11) <= 1
 0 <= control_model(3,12) <= 1
 0 <= control_model(3,13) <= 1
 0 <= control_model(3,14) <= 1
 0 <= control_model(3,15) <= 1
 0 <= control_model(3,16) <= 1
 0 <= control_model(3,17) <= 1
 0 <= control_model(3,18) <= 1
 0 <= control_model(3,19) <= 1
 0 <= control_model(4,1) <= 1
 0 <= control_model(4,2) <= 1
 0 <= control_model(4,3) <= 1
 0 <= control_model(4,4) <= 1
 0 <= control_model(4,5) <= 1
 0 <= control_model(4,6) <= 1
 0 <= control_model(4,7) <= 1
 0 <= control_model(4,8) <= 1
 0 <= control_model(4,9) <= 1
 0 <= control_model(4,10) <= 1
 0 <= control_model(4,11) <= 1
 0 <= control_model(4,12) <= 1
 0 <= control_model(4,13) <= 1
 0 <= control_model(4,14) <= 1
 0 <= control_model(4,15) <= 1
 0 <= control_model(4,16) <= 1
 0 <= control_model(4,17) <= 1
 0 <= control_model(4,18) <= 1
 0 <= control_model(4,19) <= 1
 0 <= control_model(5,1) <= 1
 0 <= control_model(5,2) <= 1
 0 <= control_model(5,3) <= 1
 0 <= control_model(5,4) <= 1
 0 <= control_model(5,5) <= 1
 0 <= control_model(5,6) <= 1
 0 <= control_model(5,7) <= 1
 0 <= control_model(5,8) <= 1
 0 <= control_model(5,9) <= 1
 0 <= control_model(5,10) <= 1
 0 <= control_model(5,11) <= 1
 0 <= control_model(5,12) <= 1
 0 <= control_model(5,13) <= 1
 0 <= control_model(5,14) <= 1
 0 <= control_model(5,15) <= 1
 0 <= control_model(5,16) <= 1
 0 <= control_model(5,17) <= 1
 0 <= control_model(5,18) <= 1
 0 <= control_model(5,19) <= 1
 abs_smoothness(1) free
 abs_smoothness(2) free
 abs_smoothness(3) free
 abs_smoothness(4) free
 abs_smoothness(5) free
 0 <= endToEnd_time(1) <= 33
 0 <= endToEnd_time(2) <= 33
 0 <= endToEnd_time(3) <= 33
 0 <= endToEnd_time(4) <= 33
 0 <= endToEnd_time(5) <= 33
 0 <= s_binary(1) <= 1
 0 <= s_binary(2) <= 1
 0 <= s_binary(3) <= 1
 0 <= s_binary(4) <= 1
 0 <= s_binary(5) <= 1

Generals
 control_model(1,1)
 control_model(1,2)
 control_model(1,3)
 control_model(1,4)
 control_model(1,5)
 control_model(1,6)
 control_model(1,7)
 control_model(1,8)
 control_model(1,9)
 control_model(1,10)
 control_model(1,11)
 control_model(1,12)
 control_model(1,13)
 control_model(1,14)
 control_model(1,15)
 control_model(1,16)
 control_model(1,17)
 control_model(1,18)
 control_model(1,19)
 control_model(2,1)
 control_model(2,2)
 control_model(2,3)
 control_model(2,4)
 control_model(2,5)
 control_model(2,6)
 control_model(2,7)
 control_model(2,8)
 control_model(2,9)
 control_model(2,10)
 control_model(2,11)
 control_model(2,12)
 control_model(2,13)
 control_model(2,14)
 control_model(2,15)
 control_model(2,16)
 control_model(2,17)
 control_model(2,18)
 control_model(2,19)
 control_model(3,1)
 control_model(3,2)
 control_model(3,3)
 control_model(3,4)
 control_model(3,5)
 control_model(3,6)
 control_model(3,7)
 control_model(3,8)
 control_model(3,9)
 control_model(3,10)
 control_model(3,11)
 control_model(3,12)
 control_model(3,13)
 control_model(3,14)
 control_model(3,15)
 control_model(3,16)
 control_model(3,17)
 control_model(3,18)
 control_model(3,19)
 control_model(4,1)
 control_model(4,2)
 control_model(4,3)
 control_model(4,4)
 control_model(4,5)
 control_model(4,6)
 control_model(4,7)
 control_model(4,8)
 control_model(4,9)
 control_model(4,10)
 control_model(4,11)
 control_model(4,12)
 control_model(4,13)
 control_model(4,14)
 control_model(4,15)
 control_model(4,16)
 control_model(4,17)
 control_model(4,18)
 control_model(4,19)
 control_model(5,1)
 control_model(5,2)
 control_model(5,3)
 control_model(5,4)
 control_model(5,5)
 control_model(5,6)
 control_model(5,7)
 control_model(5,8)
 control_model(5,9)
 control_model(5,10)
 control_model(5,11)
 control_model(5,12)
 control_model(5,13)
 control_model(5,14)
 control_model(5,15)
 control_model(5,16)
 control_model(5,17)
 control_model(5,18)
 control_model(5,19)
 s_binary(1)
 s_binary(2)
 s_binary(3)
 s_binary(4)
 s_binary(5)

End
