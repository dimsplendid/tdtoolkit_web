import path from 'path'
import express, { Request, Response } from 'express'

// routers
import { queryTotalTable } from './routes/queryTotalTable'
import { tr2Calculator } from './routes/tr2Calculator'

// template router
import { templateRouter } from './routes/template'

// test
import { extractZip, uploadZip } from './routes/lib/zipUtils'
import { update_db, calculate_summary } from './routes/lib/tr2Utils'

const app = express();
const port = 3000;

// middleware static
// make the root alias (./public -> /)
app.use(express.static(path.join(__dirname, '.', 'public')))

// template router
app.use(templateRouter)

// query table
app.use(queryTotalTable)

// TR2 calculator
app.use(tr2Calculator)

// test new upload function
app.post('/test', [
    uploadZip, 
    extractZip,
    update_db,
    calculate_summary
])

// uploads files explore?

// catch all errors
// app.use()

app.listen(port, () => {
    if (port === 3000) {
        console.log('true')
    }
    console.log(`server is listening on ${port} !!!`);
});