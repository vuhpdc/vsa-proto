#!/bin/bash
export LC_NUMERIC=en_US.UTF-8

if [ "$#" -ne 1 ]; then
  echo "Usage: ./shape <shaping file> "
  exit
fi

input_file=$1
SINK=130.37.199.8 # fs0.das5.cs.vu.nl
mapfile -t list < $input_file

#tc_adapter=wlp59s0 #wifi
# tc_adapter=enx106530c1958e #eth
tc_adapter=eth0 #eth
#tc_adapter=eno1 #server eth

TIMESTAMP=$(sleepenh 0)
# sudo tc qdisc add dev $tc_adapter root tbf rate 10mbit latency 25ms burst 3000

# Use HTB
#iperf3 -c netmsys.org -p 10001 -t 1000 &
#iperf3 -c fs0.das5.cs.vu.nl -p 10004 -t 300 &
#iperf3 -c 130.83.163.233 -p 10001 -t 1000 &

SECONDS=20

function init() {
  # echo "Initiatilizing..."
  tc qdisc add dev $tc_adapter root handle 1: htb
  tc class add dev $tc_adapter parent 1: classid 1:10 htb rate 100Mbit
  tc filter add dev $tc_adapter protocol ip parent 1: prio 1 u32 match ip dst ${SINK} flowid 1:10
}

function clear() {
  # echo "Clearing..."
  tc qdisc del dev $tc_adapter root
}

function stop() {
  # echo "Stoping..."
  clear
  exit
}

function change_bw() {
  # echo "Changing bw to $1..."
  tc class change dev $tc_adapter parent 1: classid 1:10 htb rate "$1"Mbit
}

trap "stop" SIGHUP SIGINT SIGTERM

clear
init

while true
do
  for item in ${list[@]}
  do
          if [ -z "${item//[$'\t\r\n ']}" ] #skip empty var
          then 
                  continue
          fi
          
          #sudo tc class change dev $tc_adapter parent 1: classid 1:11 htb rate "${item//[$'\t\r\n ']}"mbit
          # sudo tc qdisc change dev $tc_adapter root tbf rate "${item//[$'\t\r\n ']}"mbit latency 25ms burst 3000

          # Adjust bandwidth
          # tc class change dev $tc_adapter parent 1: classid 1:10 htb rate "${item//[$'\t\r\n ']}"mbit burst 3000
          change_bw "${item//[$'\t\r\n ']}"
        
          # Adjust latency
          # tc qdisc add dev $tc_adapter parent 1:10 handle 10: netem delay 10ms 
          
          #date +"B | %H:%M:%S.%N | $(printf "%.3f" ${item//[$'\t\r\n ']}) mbit"
          duration=$SECONDS
          # echo -n "$(printf "%.3f " ${item//[$'\t\r\n ']})"
          TIMESTAMP=$(sleepenh $TIMESTAMP $duration)
  done
done

t=$(sleepenh $t 3)

#echo "shaping completed, removing qdisc"
# tc qdisc del dev $tc_adapter root
stop
