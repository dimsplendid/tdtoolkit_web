import path from 'path'
import express from 'express'
import {Request, Response, NextFunction} from 'express'
import { PythonShell, Options } from 'python-shell'

import fs from 'fs'
import extract from 'extract-zip'
import formidable from 'formidable'

const uploadDir = path.join(__dirname, '/uploads/');
const extractDir = path.join(__dirname, '/app/');

const app = express();
const port = 3000;

// middleware static
// make the root alias (../public -> /)
app.use(express.static(path.join(__dirname, '..', 'public')))

app.get('/plot/VT', (req, res) => {
    let options = {
        mode: "text",
        scriptPath: './python_scripts',
        args: [
            req.query.LC,
            req.query.cell_gap,
            req.query.V_max,
            req.query.V_min
        ]
    } as Options
    PythonShell.run('draw_VT.py', options, (err, output) => {
        if (err) {
            res.send(err)
        } else {
            res.send(output)
        }
    })
})

// for insert vender measuring data
// Something weird here, need figure out later
// the action in "/submit/form" would just in "/submit/*" ? 

app.get('/submit/VHR', (req, res) => {
    res.send(req.query)
})
app.get('/query/total_table', (req, res) => {
    console.log(req.query)
    let options = {
        mode: "text",
        scriptPath: './python_scripts',
        args: [
            req.query.LC,
            req.query.cell_gap_lower,
            req.query.cell_gap_upper,
        ]
    } as Options
    PythonShell.run('query_total_table.py', options, (err, output) => {
        if (err) {
            res.send(err)
        } else {
            res.send(output)
        }
    })
})

// these should separate at other files later
// copy and modified from https://gist.github.com/dev-drprasad/8f46ddd8ffea7ba8f883e577d3ce0005
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);
}
if (!fs.existsSync(extractDir)) {
    fs.mkdirSync(extractDir);
}

const extractZip = (file, destination, deleteSource) => {
    extract(file, { dir: destination }, (err) => {
        if (!err) {
            if (deleteSource) fs.unlinkSync(file);
            nestedExtract(destination, extractZip);
        } else {
            console.error(err);
        }
    });
};

const nestedExtract = (dir, zipExtractor) => {
    fs.readdirSync(dir).forEach((file) => {
        if (fs.statSync(path.join(dir, file)).isFile()) {
            if (path.extname(file) === '.zip') {
                // deleteSource = true to avoid infinite loops caused by extracting same file
                zipExtractor(path.join(dir, file), dir, true);
            }
        } else {
            nestedExtract(path.join(dir, file), zipExtractor);
        }
    });
};

// there should be a better way to get file name

const uploadMedia = (req: Request, res: Response, next: NextFunction) => {
    const form = new formidable.IncomingForm();
    // file size limit 100MB. change according to your needs
    form.maxFileSize = 100 * 1024 * 1024;
    form.keepExtensions = true;
    form.multiples = true;
    form.uploadDir = uploadDir;

    // collect all form files and fileds and pass to its callback
    form.parse(req, (err, fields, files) => {
        // when form parsing fails throw error
        if (err) return res.status(500).json({ error: err });

        if (Object.keys(files).length === 0) return res.status(400).json({ message: "no files uploaded" });

        // Iterate all uploaded files and get their path, extension, final extraction path
        const filesInfo = Object.keys(files).map((key) => {
            const file = files[key];
            const filePath = file.path;
            const fileExt = path.extname(file.name);
            const fileName = path.basename(file.name, fileExt);

            return { filePath, fileExt, fileName };
        });

        // Check whether uploaded files are zip files
        const validFiles = filesInfo.every(({ fileExt }) => fileExt === '.zip');

        // if uploaded files are not zip files, return error
        if (!validFiles) return res.status(400).json({ message: "unsupported file type" });

        res.status(200).json({ uploaded: true });

        // iterate through each file path and extract them
        filesInfo.forEach(({ filePath, fileName }) => {
            // create directory with timestamp to prevent overwrite same directory names
            // const destDir = `${path.join(extractDir, fileName)}_${new Date().getTime()}`;
            const destDir = path.join(extractDir, 'raw');

            // pass deleteSource = true if source file not needed after extraction
            extractZip(filePath, destDir, false);
        });
    });

    // runs when new file detected in upload stream
    form.on('fileBegin', function (name, file) {
        // get the file base name `index.css.zip` => `index.html`
        const fileName = path.basename(file.name, path.extname(file.name));
        const fileExt = path.extname(file.name);
        // create files with timestamp to prevent overwrite same file names
        res.locals.file_name = `${fileName}_${new Date().getTime()}${fileExt}`
        // res.locals.file_name = `raw`
        file.path = path.join(uploadDir, res.locals.file_name);
    });
}
const update_db = (req: Request, res: Response, next: NextFunction) => {
    let options = {
        mode: "text",
        scriptPath: './python_scripts',
        args: [
            "test path"
        ]
    } as Options
    PythonShell.run('update_db.py', options, (err, output) => {
        if (err) {
            res.send(err)
        } else {
            res.send(output)
        }
    })
}

const calculate_summary = (req: Request, res: Response, next: NextFunction) => {
    let options = {
        mode: "text",
        scriptPath: './python_scripts',
        args: [
            req.query.batch,
        ]
    } as Options
    PythonShell.run('calculate_summary.py', options, (err, output) => {
        if (err) {
            res.send(err)
        } else {
            res.send(output)
        }
    })
}

app.post('/upload', uploadMedia);

app.get('/update_db', update_db)
app.get('/calculate_summary', calculate_summary)

app.listen(port, () => {
    if (port === 3000) {
        console.log('true')
    }
    console.log(`server is listening on ${port} !!!`);
});