import path from 'path'
import express from 'express'

// routes
import {queryTotalTable} from './routes/queryTotalTable'

// test
import {tr2Calculator} from './routes/tr2Calculator'


const app = express();
const port = 3000;

// middleware static
// make the root alias (../public -> /)
app.use(express.static(path.join(__dirname, '.', 'public')))

// query table
app.use(queryTotalTable)

// TR2 calculator
app.use(tr2Calculator)

app.listen(port, () => {
    if (port === 3000) {
        console.log('true')
    } 
    console.log(`server is listening on ${port} !!!`);
});