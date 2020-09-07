
//shorthands

//buffers
var x = [];
var y = [];
var yy= [];

var indices = cb_obj.indices;
var nIndex = indices.length;

console.log(cb_obj)
var tmp, nNodes, nTime;
var idx;

var label, labels = ['x', 'y', 'color'];
for (var i = 0; i < labels.length; i++){
    label = labels[i];
    buffer.data[label] = [];
    buffer_causal.data[label] = [];
}
var plt = Bokeh.Plotting;

console.log('start debug')
var colors_plot;
var sep;
for (var ni = 0 ; ni < nIndex ; ni++){
    // get the data; build the time vector
    idx      = indices[ni];
    nNodes   = mi.data['y'][idx].length;
    sep      = Math.floor(colors.length / nNodes); 
    
    colors_plot = [];
    for (var node = 0 ; node < nNodes ; node++){
        nTime = mi.data['y'][idx][node].length;
        // build time
        x = [];
        for (var t = 0 ; t < nTime ; t++){
            x.push(t);
        }
        colors_plot.push([colors[node * sep]])
        y = mi.data['y'][idx][node];
        buffer.data['x'].push(x);
        buffer.data['y'].push(y);
         
        yy = causal.data['y'][idx][node];
        buffer_causal.data['x'].push(x);
        buffer_causal.data['y'].push(yy);
        
        buffer_causal.data['color'].push(colors[sep * node])
        
        buffer.data['color'].push(colors[sep * node])
        }
    
    console.log(buffer.data['y'].length, colors_plot.length)
}
buffer.change.emit();
buffer_causal.change.emit();


