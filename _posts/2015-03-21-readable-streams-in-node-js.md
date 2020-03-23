---
title: 'read, _read, and readable: Readable Streams in Node.js'
date: 2015-03-21T13:29:55-04:00
layout: post
tags: ["javascript", "nodejs"]
---
One aspect of Node I had trouble wrapping my head around was how  readable streams work in the new streams2 API. After reading the [API documentation](https://nodejs.org/api/stream.html), [two](http://amzn.to/1MIFZn4) [different](http://amzn.to/1GQPdM0) books on Node, and the [stream-handbook](https://github.com/substack/stream-handbook), along with exhaustive Googling around, I still just didn’t get it. 

After spending some time with the [Node.js source](https://github.com/joyent/node), I thought I could spare others the same confusion I had. To that end, this blog post explains how readable streams work by considering the relationship between `read()`, `_read()`, and `push()` methods, and the `‘readable’` event defined on the Readable base class. 

## The Basic Relationship

Assume we’ve created a readable stream in the following way:

~~~ js
var stream = require(‘stream’);

function MyReadable () {
	stream.Readable.call(this);
	this._counter = 1;
}

util.inherits(MyReadable, stream.Readable);

MyReadable.prototype._read = function (size) {
	// This is a synchronous function
	
    this.push((this._counter++).toString());
};

var rs = new MyReadable();

var firstRead = rs.read();
var secondRead = rs.read();
~~~

The first call to `rs.read()` will trigger a call to `rs._read()`, which will in turn call `rs.push()`, which will place the argument passed to `rs.push()` into an internal buffer, which is just a simple Array. Once `rs.push()` places in the information in the internal buffer, it will cause `rs` to emit the `readable` event.

`rs.push()` then returns control to `rs._read()`, which returns control to `rs.read()`. `rs.read()` then returns all the data in the internal buffer, and clears the internal buffer in preparation for the next call.

At the end of this program, `firstRead ` will equal 1 and `secondRead` will equal 2. 

## What if rs.\_read() is asynchronous?
Now consider the situation where `_read()` is written to be an asynchronous function:

~~~ js
MyReadable.prototype._read = function (size) {
	// This is an asynchronous function
	
    setTimeout(function() {
		this.push((this._counter++).toString());
    }, 1000);
};
~~~

The first call to `rs.read()` will trigger a call to `rs._read()`, which will set a timeout and immediately return without pushing anything. 

`rs.read()` then returns all the data in the internal buffer, which at this point is empty.

The second call to `rs.read()` happens immediately, again triggering a call to `rs._read()`, and again setting a timeout, immediately returning without pushing anything. This second call to `rs.read()` returns all the data (i.e. nothing) in the still-empty internal buffer. 

When execution reaches the bottom of the program, `firstRead ` will equal `null` and `secondRead` will also equal `null`. 

Approximately one second later, both callbacks set by the two calls to `rs.read()` will fire, one following shortly after the other. The first time the callback fires, the number 1 will get pushed into the internal buffer and the `readable` event will fire. The second time it fires (likely immediately after the first one), the number 2 will get pushed into the internal buffer. 

The `readable` event will not fire after the second call to `this.push()` since the `readable` event only fires when the internal buffer goes from empty to non-empty. 

At the end of this program, the internal buffer will look something like this: `[1, 2]` and `firstRead` and `secondRead` will still both be `null`.[^1]



## What happens if the internal buffer gets filled up?

By default, the internal buffer has a limit of 16kB. The Node API calls this limit the `highWaterMark`. If data continues to be pushed into the internal buffer by calls to `push()`, but the internal buffer is never cleared, eventually, it will fill up. 

When might this happen? A lot of times, you will write readable streams to wrap other interfaces, and you will want to push data into the internal buffer when the data becomes available on the wrapped interface/source of data. For example, you could have an event handler on the source's `data` event will push the data into the readable stream's internal buffer.

But let's take a simpler example, for our purposes. Assume our program from before is the same, except the `_read` function is written in the following way:

~~~ js
MyReadable.prototype._read = function (size) {
    // Note that this is an interval, not a timeout.
    // It will fire the callback every second.
	
    if(!this.timerObject) {
	    this.timerObject = setInterval(function() {
			this.push((this._counter++).toString());
    	}, 1000);
    }
};
~~~

This pushes data into the internal buffer every second, taking care not to set multiple timers when `read()` is called more than once. 

Nothing in our program collects the data (recall that `rs.read()` is only executed twice right at the beginning of the program, and never again), so after a certain point, the `highWaterMark` will be reached.[^2]

Once the `highWaterMark` is reached, two major things happen: 

* First, `push()` will start returning `false`, although it will continue to push data onto the internal buffer above the `highWaterMark` limit. 

* Second, calls to `read()` will skip calling `_read()`. But since a call to `read()` will drain the internal buffer (bringing it back below the `highWaterMark`), subsequent calls to `read()` will again start calling `_read()`.

This second point, however, is not much help if nothing is calling `read()` to drain the internal buffer in the first place. Thus, your program must take care to stop the source data when `push()` starts returning `false`. Otherwise, the `highWaterMark` feature of readable streams is of limited usefulness. 

Here is how our example `_read()` function could be re-written:

~~~ js
MyReadable.prototype._read = function (size) {
    // Note that this is an interval, not a timeout.
    // It will fire the callback every second.
	
    if (!this.timerObject) {
	    this.timerObject = setInterval(function() {
			if (!this.push(this._counter++)) {
            	clearInterval(this.timerObject);
                this.timerObject = false;
            }
    	}, 1000);
    }
};
~~~

This would do the trick of shutting off the "source" data, which, in this case, is just an interval. The "source" data would turn back on the next time `rs._read()` is called, and, as mentioned above, this won't happen when the internal buffer has reached the `highWaterMark`.

## What is the point of the ‘readable’ event?

If the process for implementing readable streams is essentially defining a `_read()` function and calling `read()` to process data, how does the `readable` event fit into all of this? 

Truth be told, this is not a very important question. The reason it is not that important is that the most common way readable streams are used are to `pipe` from a readable stream to a writable stream. And, as we'll see in the next section, `pipe` uses something similar to the old "flow" method of streams (from the original streams API), rather than calls to the `read()` function.  

But imagine if you did not use `pipe`. When in your program would you call `read()` on the readable stream, and how would you call it repeatedly to consume the data as it became available? That is what the `readable` event is designed to address. 

You would probably write something like this:

~~~ js
rs.on('readable', function() {
	var data = rs.read();
    console.log(data.toString());
});
~~~

This would output to the console the streamed data as it came in.

One thing that might be bothering you at this point is that we agreed that `readable` is only emitted when `push()` places data in the internal buffer. And `push()` is only called if `_read()` is called, and `_read()` is only called if `read()` is called. 

But in this snippet, `read()` only gets called if there is a `readable` event, and since a `readable` event is only emitted by a chain of events set in motion by `read()`, it's as if this whole block of code does nothing.  

The trick is that the first time a listener gets registered on the `readable` event (using, e.g., `rs.on`), something like the following gets silently executed: `rs.read(0)`. This does not return any data (the 0 represents that read should return no data), but it kickstarts the chain of events.

One thing to be aware of is that `readable` is only emitted when data is pushed to the internal buffer at a moment when the internal buffer previously had no data in it (it was "drained"). Although this is not much of a consideration if `read()` is being called with no arguments, since such a call will attempt to clear and return the entire internal buffer.[^3] 

## How does rs.pipe() work?

As mentioned previously, `pipe()` is a common way streams are used. You'll commonly see something like:

~~~ js
readable.pipe(writable);
~~~
The bad news is that most of what we've discussed so far is of limited usefulness when it comes to `pipe()`, which uses the "flow" mode of streams that resembles the original streams API of Node. 

A fulsome discussion of "flow" mode is outside the scope of this article, but in short, `pipe()` does not make individual calls `read()`, but rather listens for the `data` event and feeds that data into a writable stream, pausing the readable stream when it detects backpressure from the writeable stream. 

Although `pipe()` works somewhat differently than the methods outlined in this article, your job when implementing a readable stream is largely the same. You still need only define the `_read()` function that calls `push()` to add data into the internal buffer, taking the same care to handle the situation of when the `highWaterMark` is reached (and `push()` starts returning `false`). 

Finally, one additional consideration when planning for the use of `pipe()` with your readable stream is to call `push(null)` when the underlying source has no more data to provide. This will cause the readable stream to emit `end`, which `pipe()` will use to automatically call `end()` on the _writeable_ stream. 

## Conclusion

Armed with this knowledge, a read through the [Node streams API documentation](https://nodejs.org/api/stream.html) as well as the publicly available [stream handbook](https://github.com/substack/stream-handbook) could be more useful. And, if you are feeling adventurous, the [Node source code](https://github.com/joyent/node) implementing readable streams is well-commented. 

## Footnotes
[^1]: This is not strictly true, but I'm presenting it this way for pedagogical reasons. The truth is that the program won't actually end -- it will keep running in perpetuity. The reason is that if `push()` detects that nothing is causing read() to continue to be called, it will call `read(0)` before returning. `read(0)` will not return any data, but it will trigger another call to `_read()`, which will, in this example, register another one second timeout. One second later, the callback to the timeout will fire, and this cycle will start again.
[^2]: Calculating how many hours the program will take to reach the `highWaterMark` is left as an exercise for the reader.
[^3]: If `read()` is called with arguments (e.g., `rs.read(5)`), it will attempt to read and return only a fixed amount of data from the internal buffer. A look at how this process works is outside the scope of this discussion.
