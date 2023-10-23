const { app, BrowserWindow } = require("electron");
const path = require("path");
const {spawn} = require("child_process")
const { get } = require('axios');
const kill = require('tree-kill')

let pythonServer;

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require("electron-squirrel-startup")) {
	app.quit();
}

const createWindow = () => {
	// Create the browser window.
	const mainWindow = new BrowserWindow({
		width: 800,
		height: 600,
		webPreferences: {
			preload: MAIN_WINDOW_PRELOAD_WEBPACK_ENTRY,
		},
	});

	// and load the index.html of the app.
	mainWindow.loadURL(MAIN_WINDOW_WEBPACK_ENTRY);

	// Open the DevTools.
	mainWindow.webContents.openDevTools();

  if (app.isPackaged) {
    const runFlask = {
      darwin: `open -gj "${path.join(process.resourcesPath, 'resources', 'app.app')}" --args`,
      linux: `${path.join(process.resourcesPath, "app", "app")}`,
      win32: `start ${path.join(process.resourcesPath, "app", "app.exe")}`,
      // linux: './resources/app/app',
      // win32: 'start ./resources/app/app.exe'
    }[process.platform];
  
    pythonServer = spawn(`${runFlask}`, { detached: false, shell: true, stdio: 'pipe' });
  } else {
    pythonServer = spawn(`python app.py`, { detached: true, shell: true, stdio: 'inherit' });
  }
};

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on("ready", () => {
	createWindow();
  // console.log(pythonServer)
  // kill(pythonServer.pid)
  // console.log(pythonServer)
});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on("window-all-closed", () => {
	if (process.platform !== "darwin") {
    // get(`http://localhost:5000/quit`)
    // .then(app.quit)
    // .catch(app.quit);
    // console.log("QUITTING")
    kill(pythonServer.pid)
		app.quit();
	}
});

let pythonServerKilled = false;

app.on("before-quit", (event) => {
  if (!pythonServerKilled) {
    event.preventDefault();
    console.log("KILLING SERVER!")
    kill(pythonServer.pid, () => {
      pythonServerKilled = true;
      app.quit();
    });
    // pythonServerKilled = true;
    // app.quit();
  }
})

app.on("activate", () => {
	// On OS X it's common to re-create a window in the app when the
	// dock icon is clicked and there are no other windows open.
	if (BrowserWindow.getAllWindows().length === 0) {
		createWindow();
	}
});

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and import them here.
