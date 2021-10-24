for f in *png; do
  convert $f -crop 1000x820+120+40 $f
done
