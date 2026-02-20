$NodeDir = "c:\temp\geoportal\proyecto_logai\tools\node-v20.11.0-win-x64"
$env:Path = "$NodeDir;" + $env:Path
cd logai-front
npm start
