import path from 'path'
import fs from 'fs'
import extract from 'extract-zip'
import formidable from 'formidable'
import express from 'express'
import {Request, Response, NextFunction} from 'express'
import { PythonShell, Options } from 'python-shell'

const router = express.Router()

const uploadDir = path.join(__dirname, '/uploads/');
const extractDir = path.join(__dirname, '/extract_files/');

// copy and modified from https://gist.github.com/dev-drprasad/8f46ddd8ffea7ba8f883e577d3ce0005
// need modifed, cause I shouldn't need recursive un-archive.

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
    if (!fs.existsSync(uploadDir)) {
        fs.mkdirSync(uploadDir);
    }
    if (!fs.existsSync(extractDir)) {
        fs.mkdirSync(extractDir);
    }
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

        // res.status(200).json({ uploaded: true });

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
    next();
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
            // res.send(output[output?.length - 1])
            req.batch = output[output?.length - 1]
            next()
        }
    })
}

const calculate_summary = (req: Request, res: Response, next: NextFunction) => {
    let options = {
        mode: "text",
        scriptPath: './python_scripts',
        args: [
            // req.query.batch,
            req.batch
        ]
    } as Options
    PythonShell.run('calculate_summary.py', options, (err, output) => {
        if (err) {
            res.send(err)
        } else {
            let file_name = output[output?.length - 1] // need modified later
            res.download(file_name)
        }
    })
}

router.post('/upload', uploadMedia, update_db, calculate_summary);

export {router as tr2Calculator}