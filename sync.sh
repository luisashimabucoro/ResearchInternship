#! /bin/bash
# sync
# usage: sync.sh project 

help () { 
   echo "usage: sync.sh [options] SERVER PROJECT_NAME"
   echo
   echo "Options:"
   echo "   -u <server_username>"
   echo "   -s <server_hostname>"
   echo "   -p <port>"
   echo "   -d <server_projects_folder>"
   echo "   -l <local_projects_folder>"
   echo "   -f <file_pattern>"
   echo "   -e <gitignore-like_exclude_file>"
}

# check for correct usage
if test $# -lt 2
then
    help
    exit 0
fi

# default values
server_username="s2589574"
server_hostname="staff.ssh.inf.ed.ac.uk"
server_projects_folder="/afs/inf.ed.ac.uk/user/s25/s2589574/"
local_projects_folder="/home/lushimabucoro/Codes/ResearchInternship"
uses_file_pattern=false
file_pattern=""
# exclude_file=".gitignore"
exclude_file=""
is_surrey=0

# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

while getopts "h?u:s:p:d:l:f:e:" opt; do
    case "$opt" in
    h|\?)
        help
        exit 0
        ;;
    u)  
        server_username=$OPTARG
        ;;
    s)  
        server_hostname=$OPTARG
        ;;
    p)  
        port=$OPTARG
        ;;
    d)  
        server_projects_folder=$OPTARG
        ;;
    l)  
        local_projects_folder=$OPTARG
        ;;
    f)  
        file_pattern=$OPTARG
        ;;
    f)  
        exclude_file=$OPTARG
        ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
    esac
done

shift $((OPTIND-1))

[ "$1" = "--" ] && shift

server=$1
project=$2

case "$server" in
        invincible)
            server_username="s2589574"
            server_hostname="staff.ssh.inf.ed.ac.uk"
            ;;
        esac

exclude_statement=""

if [ -e $local_projects_folder/$project/$exclude_file ]; then
    /bin/rsync --exclude-from "/$local_projects_folder/$project/$exclude_file" -avzhe "ssh" /$local_projects_folder/$project/$file_pattern $server_username@$server_hostname:$server_projects_folder/$project
else
    /bin/rsync -avzhe "ssh" /$local_projects_folder/$project/$file_pattern $server_username@$server_hostname:$server_projects_folder/$project
fi

# /usr/local/bin/rsync