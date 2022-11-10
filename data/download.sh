echo "downloading..."
svn export https://github.com/recski/brise-plandok/trunk/brise_plandok/baselines/input csv_files
echo "folder downloaded"
echo "extracting files..."
cd csv_files || exit
for filename in * ; do
  mv "$filename" "../"
done

cd "../" || exit
rm -r csv_files
echo "done"