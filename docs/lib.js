/*

This file contains only functions necessary for the article features
The full library code and enhanced versions of the functions present
here can be found at http://v2studio.com/k/code/lib/


ARRAY EXTENSIONS

push(item [,...,item])
    Mimics standard push for IE5, which doesn't implement it.


find(value [, start])
    searches array for value starting at start (if start is not provided,
    searches from the beginning). returns value index if found, otherwise
    returns -1;


has(value)
    returns true if value is found in array, otherwise false;


FUNCTIONAL

map(list, func)
    traverses list, applying func to list, returning an array of values returned
    by func

    if func is not provided, the array item is returned itself. this is an easy
    way to transform fake arrays (e.g. the arguments object of a function or
    nodeList objects) into real javascript arrays.

    map also provides a safe way for traversing only an array's indexed items,
    ignoring its other properties. (as opposed to how for-in works)

    this is a simplified version of python's map. parameter order is different,
    only a single list (array) is accepted, and the parameters passed to func
    are different:
    func takes the current item, then, optionally, the current index and a
    reference to the list (so that func can modify list)


filter(list, func)
    returns an array of values in list for which func is true

    if func is not specified the values are evaluated themselves, that is,
    filter will return an array of the values in list which evaluate to true

    this is a similar to python's filter, but parameter order is inverted


DOM

getElem(elem)
    returns an element in document. elem can be the id of such element or the
    element itself (in which case the function does nothing, merely returning
    it)

    this function is useful to enable other functions to take either an    element
    directly or an element id as parameter.

    if elem is string and there's no element with such id, it throws an error.
    if elem is an object but not an Element, it's returned anyway


hasClass(elem, className)
    Checks the class list of element elem or element of id elem for className,
    if found, returns true, otherwise false.

    The tested element can have multiple space-separated classes. className must
    be a single class (i.e. can't be a list).


getElementsByClass(className [, tagName [, parentNode]])
    Returns elements having class className, optionally being a tag tagName
    (otherwise any tag), optionally being a descendant of parentNode (otherwise
    the whole document is searched)


DOM EVENTS

listen(event,elem,func)
    x-browser function to add event listeners

    listens for event on elem with func
    event is string denoting the event name without the on- prefix. e.g. 'click'
    elem is either the element object or the element's id
    func is the function to call when the event is triggered

    in IE, func is wrapped and this wrapper passes in a W3CDOM_Event (a faux
    simplified Event object)


mlisten(event, elem_list, func)
    same as listen but takes an element list (a NodeList, Array, etc) instead of
    an element.


W3CDOM_Event(currentTarget)
    is a faux Event constructor. it should be passed in IE when a function
    expects a real Event object. For now it only implements the currentTarget
    property and the preventDefault method.

    The currentTarget value must be passed as a paremeter at the moment    of
    construction.


MISC CLEANING-AFTER-MICROSOFT STUFF

isUndefined(v)
    returns true if [v] is not defined, false otherwise

    IE 5.0 does not support the undefined keyword, so we cannot do a direct
    comparison such as v===undefined.
*/

// ARRAY EXTENSIONS

if (!Array.prototype.push) Array.prototype.push = function() {
    for (var i=0; i<arguments.length; i++) this[this.length] = arguments[i];
    return this.length;
}

Array.prototype.find = function(value, start) {
    start = start || 0;
    for (var i=start; i<this.length; i++)
        if (this[i]==value)
            return i;
    return -1;
}

Array.prototype.has = function(value) {
    return this.find(value)!==-1;
}

// FUNCTIONAL

function map(list, func) {
    var result = [];
    func = func || function(v) {return v};
    for (var i=0; i < list.length; i++) result.push(func(list[i], i, list));
    return result;
}

function filter(list, func) {
    var result = [];
    func = func || function(v) {return v};
    map(list, function(v) { if (func(v)) result.push(v) } );
    return result;
}


// DOM

function getElem(elem) {
    if (document.getElementById) {
        if (typeof elem == "string") {
            elem = document.getElementById(elem);
            if (elem===null) throw 'cannot get element: element does not exist';
        } else if (typeof elem != "object") {
            throw 'cannot get element: invalid datatype';
        }
    } else throw 'cannot get element: unsupported DOM';
    return elem;
}

function hasClass(elem, className) {
    return getElem(elem).className.split(' ').has(className);
}

function getElementsByClass(className, tagName, parentNode) {
    parentNode = !isUndefined(parentNode)? getElem(parentNode) : document;
    if (isUndefined(tagName)) tagName = '*';
    return filter(parentNode.getElementsByTagName(tagName),
        function(elem) { return hasClass(elem, className) });
}


// DOM EVENTS

function listen(event, elem, func) {
    elem = getElem(elem);
    if (elem.addEventListener)  // W3C DOM
        elem.addEventListener(event,func,false);
    else if (elem.attachEvent)  // IE DOM
        elem.attachEvent('on'+event, function(){ func(new W3CDOM_Event(elem)) } );
        // for IE we use a wrapper function that passes in a simplified faux Event object.
    else throw 'cannot add event listener';
}

function mlisten(event, elem_list, func) {
    map(elem_list, function(elem) { listen(event, elem, func) } );
}

function W3CDOM_Event(currentTarget) {
    this.currentTarget  = currentTarget;
    this.preventDefault = function() { window.event.returnValue = false }
    return this;
}


// MISC CLEANING-AFTER-MICROSOFT STUFF

function isUndefined(v) {
    var undef;
    return v===undef;
}

