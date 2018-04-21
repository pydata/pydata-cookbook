#convert MD to rst format and save to .txt for the main .rst include
pandoc --from=markdown --to=rst --output=${1}.txt ${1}.md
