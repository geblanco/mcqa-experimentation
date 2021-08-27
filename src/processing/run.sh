#!/bin/bash

DONT_DOCKERIZE=1

fix_experiment_path(){
  local exp=$1
  if [[ " $(basename $1) " =~ " ${exp} " ]]; then
    # just file name
    echo "experiments/${exp}"
  else
    echo "${exp}"
  fi
}

run_experiment(){
  local file=$1; shift
  local script_file="mc_transformers"
  if [[ ! -z ${DONT_DOCKERIZE} ]]; then
    inside_docker=""
  else
    inside_docker="nvidia-docker run --rm ${docker_args[@]}"
  fi
  echo "$PWD"
  ${inside_docker} ${script_file} $(python3 $file_as_args -f $file)
}

get_experiments(){
  local args=($@);
  local aux_flist=()
  experiments=()
  for arg in "${args[@]}"; do
    # if it is a filelist, parse it
    if [[ " ${arg##*.} " =~ " filelist " ]]; then
      IFS=$'\n' read -d '' -r -a aux_flist < $arg
      echo "* Read experiments from $arg:"
      echo "* ${aux_flist[@]}"
      for aux in "${aux_flist[@]}"; do 
        experiments+=($aux)
      done
    else
      experiments+=($arg)
    fi
  done
}

ch_to_project_root(){
  # chdir to project root
  cd $(dvc root) >/dev/null
}

ch_to_project_root
docker_img="quail-experiments-v2"
docker_args="--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`:/workspace -v /data:/data $docker_img"
file_as_args="./src/processing/file_to_program_args.py"

echo "###### Starting experiments $(date)"
total_start_time=$(date -u +%s)
# echo $(python3 $file_as_args $@)
mc_transformers $(python3 $file_as_args $@ -x model_type)
total_end_time=$(date -u +%s)
total_elapsed=$(python3 -c "print('{:.2f}'.format(($total_end_time - $total_start_time)/60.0 ))")
echo "###### End of experiments $(date) ($total_elapsed) minutes"

