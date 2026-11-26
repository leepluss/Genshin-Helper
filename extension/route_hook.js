(function () {
    'use strict';

    if (!window.location.hostname.includes('hoyolab.com') || !window.location.pathname.includes('/interactive-map/')) {
        return;
    }
    if (window.__ghRouteHookVer === 3) {
        return;
    }
    window.__ghRouteHookVer = 3;

    const prevBuf = (window.__ghRouteHookBuffer && typeof window.__ghRouteHookBuffer === 'object')
        ? window.__ghRouteHookBuffer
        : {};
    const buf = {
        recommends: Array.isArray(prevBuf.recommends) ? prevBuf.recommends : [],
        routesById: (prevBuf.routesById && typeof prevBuf.routesById === 'object') ? prevBuf.routesById : {},
        lastRoute: prevBuf.lastRoute || null
    };
    window.__ghRouteHookBuffer = buf;

    function normalizeUrl(raw) {
        if (typeof raw === 'string') return raw;
        if (raw && typeof raw.url === 'string') return raw.url;
        if (raw && typeof raw.href === 'string') return raw.href;
        try { return raw ? String(raw) : ''; } catch (e) { return ''; }
    }

    function isRouteApiUrl(urlStr) {
        const u = normalizeUrl(urlStr);
        if (!u) return false;
        const low = u.toLowerCase();
        if (
            low.includes('point_list') ||
            low.includes('/point/list') ||
            low.includes('label_list') ||
            low.includes('/label/tree') ||
            low.includes('mark_map_point') ||
            low.includes('/map/anchor') ||
            low.includes('/map/info') ||
            low.includes('/point/info')
        ) {
            return false;
        }
        if (low.includes('get_recommend_path') || low.includes('get_share_map_spots')) {
            return true;
        }
        return low.includes('/map_user/') && (
            low.includes('path') ||
            low.includes('route') ||
            low.includes('spot') ||
            low.includes('share') ||
            low.includes('line') ||
            low.includes('plan')
        );
    }

    function extractShareId(urlStr) {
        const u = normalizeUrl(urlStr);
        if (!u) return '';
        const m = u.match(/[?&#](?:share_id|path_id|obj_id)=([^&#]+)/i);
        if (!m || !m[1]) return '';
        try { return decodeURIComponent(m[1]).trim(); } catch (e) { return String(m[1]).trim(); }
    }

    function extractPathObj(d) {
        if (!d || typeof d !== 'object') return null;
        if (d.path && typeof d.path === 'object' && Array.isArray(d.path.sections) && d.path.sections.length > 0) {
            return d.path;
        }
        if (Array.isArray(d.sections) && d.sections.length > 0) {
            return d;
        }
        if (d.route && typeof d.route === 'object' && Array.isArray(d.route.sections) && d.route.sections.length > 0) {
            return d.route;
        }
        if (d.info && typeof d.info === 'object' && Array.isArray(d.info.sections) && d.info.sections.length > 0) {
            return d.info;
        }
        return null;
    }

    function storeRouteInBuf(key, entry) {
        if (!key) return;
        const k = String(key).trim();
        if (!k) return;
        buf.routesById[k] = entry;
        const keys = Object.keys(buf.routesById);
        if (keys.length > 32) {
            keys.sort((a, b) => (buf.routesById[a]?.ts || 0) - (buf.routesById[b]?.ts || 0));
            for (let i = 0; i < keys.length - 32; i++) {
                delete buf.routesById[keys[i]];
            }
        }
    }

    function dispatchHookedJson(urlStr, jsonData) {
        try {
            if (!jsonData || typeof jsonData !== 'object') return;
            const d = jsonData.data;
            if (!d || typeof d !== 'object') return;
            const low = normalizeUrl(urlStr).toLowerCase();

            if (Array.isArray(d.list)) {
                const looksLikeRouteList = low.includes('get_recommend_path') ||
                    low.includes('path') ||
                    low.includes('route') ||
                    d.list.some((it) => it && typeof it === 'object' && (it.path_id || it.path_name || Array.isArray(it.sections)));
                if (looksLikeRouteList) {
                    buf.recommends.push({ url: urlStr, data: jsonData, ts: Date.now() });
                    while (buf.recommends.length > 8) {
                        buf.recommends.shift();
                    }
                    for (const item of d.list) {
                        if (!item || typeof item !== 'object') continue;
                        const nested = extractPathObj(item);
                        if (nested) {
                            const pid = String(item.path_id || item.obj_id || nested.obj_id || nested.path_id || '');
                            const wrapped = { data: { path: nested } };
                            if (pid) {
                                storeRouteInBuf(pid, { url: urlStr, data: wrapped, ts: Date.now() });
                            }
                        }
                    }
                    if (typeof window.__ghOnHoyolabRouteApi === 'function') {
                        window.__ghOnHoyolabRouteApi(urlStr, jsonData);
                    }
                    return;
                }
            }

            const pathObj = extractPathObj(d);
            if (pathObj) {
                const entry = { url: urlStr, data: jsonData, ts: Date.now() };
                buf.lastRoute = entry;
                const pid = String(pathObj.obj_id || pathObj.path_id || '');
                if (pid) storeRouteInBuf(pid, entry);
                const sid = extractShareId(urlStr);
                if (sid && sid !== pid) storeRouteInBuf(sid, entry);
                if (typeof window.__ghOnHoyolabRouteApi === 'function') {
                    window.__ghOnHoyolabRouteApi(urlStr, jsonData);
                }
            }
        } catch (e) {}
    }

    if (typeof window.fetch === 'function') {
        const origFetch = window.fetch;
        window.fetch = function (...args) {
            const urlStr = normalizeUrl(args[0]);
            const p = origFetch.apply(this, args);
            if (isRouteApiUrl(urlStr)) {
                p.then((res) => {
                    if (!res || !res.ok || typeof res.clone !== 'function') return;
                    res.clone().json()
                        .then((j) => dispatchHookedJson(urlStr, j))
                        .catch(() => {});
                }).catch(() => {});
            }
            return p;
        };
    }

    if (window.XMLHttpRequest && window.XMLHttpRequest.prototype) {
        const origOpen = window.XMLHttpRequest.prototype.open;
        const origSend = window.XMLHttpRequest.prototype.send;
        window.XMLHttpRequest.prototype.open = function (method, url, ...rest) {
            this.__ghReqUrl = normalizeUrl(url);
            return origOpen.call(this, method, url, ...rest);
        };
        window.XMLHttpRequest.prototype.send = function (...args) {
            const u = this.__ghReqUrl || '';
            if (isRouteApiUrl(u)) {
                this.addEventListener('load', () => {
                    try {
                        if (this.status >= 200 && this.status < 300) {
                            let payload = null;
                            if (this.responseType === 'json') {
                                payload = this.response;
                            } else if (!this.responseType || this.responseType === 'text') {
                                payload = (typeof this.responseText === 'string' && this.responseText)
                                    ? JSON.parse(this.responseText)
                                    : null;
                            } else if (this.responseType === 'arraybuffer' && this.response && typeof TextDecoder === 'function') {
                                const txt = new TextDecoder('utf-8').decode(new Uint8Array(this.response));
                                payload = txt ? JSON.parse(txt) : null;
                            }
                            if (payload) {
                                dispatchHookedJson(u, payload);
                            }
                        }
                    } catch (e) {}
                }, { once: true });
            }
            return origSend.apply(this, args);
        };
    }
})();
