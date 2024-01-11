for i in {1..20}; do
  tar czvf "autonomous_roboset_robopen05_"$i".tar.gz" $(ls /mnt/raid5/data/jaydv/autonomous_bin_pick/set/*.h5 | head -1000)
  $(ls /mnt/raid5/data/jaydv/autonomous_bin_pick/set/*.h5 | head -1000) >> autonomous_roboset_robopen05_"$i".txt
  rm $(ls /mnt/raid5/data/jaydv/autonomous_bin_pick/set/*.h5 | head -1000)
done