# generated from colcon_powershell/shell/template/prefix_chain.ps1.em

# This script extends the environment with the environment of other prefix
# paths which were sourced when this file was generated as well as all packages
# contained in this prefix path.

# function to source another script with conditional trace output
# first argument: the path of the script
function _colcon_prefix_chain_powershell_source_script {
  param (
    $_colcon_prefix_chain_powershell_source_script_param
  )
  # source script with conditional trace output
  if (Test-Path $_colcon_prefix_chain_powershell_source_script_param) {
    if ($env:COLCON_TRACE) {
      echo ". '$_colcon_prefix_chain_powershell_source_script_param'"
    }
    . "$_colcon_prefix_chain_powershell_source_script_param"
  } else {
    Write-Error "not found: '$_colcon_prefix_chain_powershell_source_script_param'"
  }
}

# source chained prefixes
_colcon_prefix_chain_powershell_source_script "/opt/ros/galactic\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/ishan/Downloads/robotarm_ws/install\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/ishan/enpm663_ws/install\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/ishan/ariac_ws/install\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/ishan/Documents/UMD/fall23/ENPM662-Modeling/Assignments/Project0/testWS/install\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/ishan/ros2_ws/install\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/ishan/enpm662_ws/install\local_setup.ps1"

# source this prefix
$env:COLCON_CURRENT_PREFIX=(Split-Path $PSCommandPath -Parent)
_colcon_prefix_chain_powershell_source_script "$env:COLCON_CURRENT_PREFIX\local_setup.ps1"
