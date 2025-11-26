// User front-end: faster verify, correct age on Poli, progress UI.
(() => {
  // Pages
  const pageHome=document.getElementById('page-home');
  const pageRegistrasi=document.getElementById('page-registrasi');
  const pagePoli=document.getElementById('page-poli');
  const pagePoliGateway=document.getElementById('page-poli-gateway');

  // Nav
  const navHome=document.getElementById('nav-home');
  const navRegistrasi=document.getElementById('nav-registrasi');
  const navPoli=document.getElementById('nav-poli');
  const btnHomeRegistrasi=document.getElementById('btn-home-registrasi');
  const btnHomeKePoli=document.getElementById('btn-home-ke-poli');

  // Registrasi
  const formRegistrasi=document.getElementById('form-registrasi');
  const inputNik=document.getElementById('reg-nik');
  const inputNama=document.getElementById('reg-nama');
  const inputDob=document.getElementById('reg-ttl');
  const inputAlamat=document.getElementById('reg-alamat');
  const videoReg=document.getElementById('video-reg');
  const statusReg=document.getElementById('status-reg');
  const countReg=document.getElementById('count-reg');

  // Verifikasi
  const videoVerif=document.getElementById('video-verif');
  const btnScan=document.getElementById('btn-scan');
  const btnNikFallback=document.getElementById('btn-nik-fallback');
  const verifResult=document.getElementById('verif-result');
  const verifData=document.getElementById('verif-data');
  const verifNikBox=document.getElementById('verif-nik');
  const fallbackNik=document.getElementById('fallback-nik');
  const btnCariNik=document.getElementById('btn-cari-nik');
  const statusVerif=document.getElementById('status-verif');
  const btnLanjutForm=document.getElementById('btn-lanjut-form');
  const btnDetailData=document.getElementById('btn-detail-data');

  // Poli gateway
  const formPoliGateway=document.getElementById('form-poli-gateway');
  const gwNama=document.getElementById('gw-nama');
  const gwUmur=document.getElementById('gw-umur');
  const gwAlamat=document.getElementById('gw-alamat');
  const gwPoli=document.getElementById('gw-poli');

  // Modals
  const modalAlert=document.getElementById('modal-alert');
  const alertMessage=document.getElementById('alert-message');
  const btnAlertOk=document.getElementById('btn-modal-alert-ok');

  const modalLoading=document.getElementById('modal-loading');
  const loadingText=document.getElementById('loading-text');
  const progressInner=document.getElementById('progress-inner');

  const modalAntrian=document.getElementById('modal-antrian');
  const antrianPoli=document.getElementById('antrian-poli');
  const antrianNomor=document.getElementById('antrian-nomor');
  const btnAntrianTutup=document.getElementById('btn-modal-antrian-tutup');

  const modalRegisSuccess=document.getElementById('modal-regis-success');
  const btnModalRegisTutup=document.getElementById('btn-modal-regis-tutup');
  const btnModalLanjutPoli=document.getElementById('btn-modal-lanjut-poli');

  const modalVerifDetail=document.getElementById('modal-verif-detail');
  const btnModalVerifTutup=document.getElementById('btn-modal-verif-tutup');
  const btnModalVerifCloseX=document.getElementById('btn-modal-verif-close-x');
  const btnModalVerifLanjut=document.getElementById('btn-modal-verif-lanjut');
  const detailPasienContent=document.getElementById('detail-pasien-content');

  // State
  let activePatient=null;
  let streamReg=null;
  let streamVerif=null;

  // Helpers
  function showPage(id){
    [pageHome,pageRegistrasi,pagePoli,pagePoliGateway].forEach(p=>p.classList.add('hidden'));
    document.querySelectorAll('.nav-button').forEach(b=>b.classList.remove('active'));
    if(id==='page-home'){pageHome.classList.remove('hidden');}
    if(id==='page-registrasi'){pageRegistrasi.classList.remove('hidden');navRegistrasi.classList.add('active');ensureCamera('reg');}
    if(id==='page-poli'){pagePoli.classList.remove('hidden');navPoli.classList.add('active');ensureCamera('verif');resetVerif();}
    if(id==='page-poli-gateway'){pagePoliGateway.classList.remove('hidden');navPoli.classList.add('active');}
  }
  function resetVerif(){
    verifResult.classList.add('hidden');
    verifNikBox.classList.add('hidden');
    statusVerif.textContent='Menunggu...';
    verifData.innerHTML='';
  }
  function showAlert(msg){alertMessage.textContent=msg;modalAlert.classList.remove('hidden');}
  function hideAlert(){modalAlert.classList.add('hidden');}
  function showLoading(text){loadingText.textContent=text;progressInner.style.width='0%';modalLoading.classList.remove('hidden');}
  function hideLoading(){modalLoading.classList.add('hidden');}
  function updateProgress(c,t,label){
    const pct=Math.round((c/t)*100);
    progressInner.style.width=pct+'%';
    loadingText.textContent=`${label} ${c}/${t} (${pct}%)`;
  }
  function openAntrian(poli,nomor){antrianPoli.textContent=`Poli: ${poli}`;antrianNomor.textContent=nomor;modalAntrian.classList.remove('hidden');}
  function closeAntrian(){modalAntrian.classList.add('hidden');}

  function computeAge(dob){
    // dob: YYYY-MM-DD
    try{
      const d=new Date(dob);
      if (isNaN(d.getTime())) return '–';
      const today=new Date();
      let age=today.getFullYear()-d.getFullYear();
      const m=today.getMonth()-d.getMonth();
      if (m<0||(m===0 && today.getDate()<d.getDate())) age--;
      return `${age} Tahun`;
    }catch(_){return '–';}
  }

  async function initWebcam(videoEl){
    try{
      const stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
      videoEl.srcObject=stream;
      return stream;
    }catch(e){
      showAlert('Gagal akses webcam: '+e.message);
      return null;
    }
  }
  async function ensureCamera(mode){
    if(mode==='reg'&&!streamReg)streamReg=await initWebcam(videoReg);
    if(mode==='verif'&&!streamVerif)streamVerif=await initWebcam(videoVerif);
  }

  function captureFrames(videoEl,total=20,gap=120,counterEl=null,label='Frame',quality=0.85){
    return new Promise(resolve=>{
      const canvas=document.createElement('canvas');
      const ctx=canvas.getContext('2d');
      const frames=[];let taken=0;
      const grab=()=>{
        if(!videoEl.videoWidth)return requestAnimationFrame(grab);
        canvas.width=videoEl.videoWidth;canvas.height=videoEl.videoHeight;
        ctx.drawImage(videoEl,0,0);
        canvas.toBlob(b=>{
          frames.push(b);taken++;
          if(counterEl)counterEl.textContent=taken;
          updateProgress(taken,total,label);
          if(taken>=total)resolve(frames);
          else setTimeout(grab,gap);
        },'image/jpeg',quality);
      };
      grab();
    });
  }

  // Registrasi (20 frame tetap, dataset bagus)
  formRegistrasi.addEventListener('submit',async e=>{
    e.preventDefault();
    const nikVal=inputNik.value.trim();
    if(!/^\d{16}$/.test(nikVal)){
      showAlert('NIK harus 16 digit angka.');
      return;
    }
    await ensureCamera('reg');
    if(!streamReg)return;
    statusReg.textContent='Mengambil foto...'; countReg.textContent='0';
    showLoading('Registrasi: mengambil foto...');
    const frames=await captureFrames(videoReg,20,120,countReg,'Foto',0.85);
    updateProgress(20,20,'Mengirim'); statusReg.textContent='Mengirim...';

    const fd=new FormData();
    fd.append('nik',nikVal);
    fd.append('name',inputNama.value.trim());
    fd.append('dob',inputDob.value);
    fd.append('address',inputAlamat.value.trim());
    frames.forEach((b,i)=>fd.append('frames[]',b,`frame_${i}.jpg`));
    try{
      const r=await fetch('/api/register',{method:'POST',body:fd});
      const d=await r.json();
      hideLoading();
      if(!d.ok){showAlert(d.msg||'Registrasi gagal');statusReg.textContent='Gagal';return;}
      statusReg.textContent='Berhasil';
      activePatient={nik:nikVal,name:inputNama.value.trim(),address:inputAlamat.value.trim(),dob:inputDob.value};
      // Clear form UI
      formRegistrasi.reset(); countReg.textContent='0';
      modalRegisSuccess.classList.remove('hidden');
    }catch(err){
      hideLoading(); showAlert('Error jaringan: '+err.message); statusReg.textContent='Error';
    }
  });

  btnModalRegisTutup.addEventListener('click',()=>{modalRegisSuccess.classList.add('hidden');showPage('page-home');});
  btnModalLanjutPoli.addEventListener('click',()=>{
    modalRegisSuccess.classList.add('hidden');
    if(activePatient){
      gwNama.textContent=activePatient.name;
      gwUmur.textContent=computeAge(activePatient.dob); // FIX umur terisi otomatis
      gwAlamat.textContent=activePatient.address;
      showPage('page-poli-gateway');
    } else showAlert('Data pasien tidak tersedia.');
  });

  // Verifikasi
  btnScan.addEventListener('click',async ()=>{
    await ensureCamera('verif'); if(!streamVerif) return;
    statusVerif.textContent='Memverifikasi...';
    showLoading('Verifikasi: mengambil foto...');
    const frames=await captureFrames(videoVerif,20,100,null,'Verifikasi',0.80);
    updateProgress(20,20,'Memproses');
    const fd=new FormData(); frames.forEach((b,i)=>fd.append('frames[]',b,`scan_${i}.jpg`));
    try{
      const r=await fetch('/api/recognize',{method:'POST',body:fd});
      const d=await r.json(); hideLoading();
      if(!d.ok){showAlert(d.msg||'Verifikasi gagal');statusVerif.textContent='Gagal';return;}
      if(!d.found){statusVerif.textContent='Tidak dikenali';showAlert(d.msg||'Wajah tidak dikenali.');activePatient=null;verifResult.classList.add('hidden');return;}
      statusVerif.textContent='Berhasil';
      activePatient={nik:d.nik,name:d.name,address:d.address,dob:d.dob,age:d.age,confidence:d.confidence};
      verifData.innerHTML=`
        <p><strong>NIK:</strong> <span class="font-mono">${d.nik}</span></p>
        <p><strong>Nama:</strong> ${d.name}</p>
        <p><strong>Umur:</strong> ${d.age}</p>
        <p><strong>Alamat:</strong> ${d.address}</p>
        <p><strong>Tingkat Kecocokan:</strong> ${d.confidence}%</p>
      `;
      verifResult.classList.remove('hidden');
    }catch(err){
      hideLoading(); showAlert('Error jaringan: '+err.message); statusVerif.textContent='Error';
    }
  });

  // Detail Data → ke Poli
  btnDetailData.addEventListener('click',()=>{
    if(!activePatient){showAlert('Tidak ada data pasien.');return;}
    detailPasienContent.innerHTML=`
      <p><strong>NIK:</strong> <span class="font-mono">${activePatient.nik}</span></p>
      <p><strong>Nama:</strong> ${activePatient.name}</p>
      <p><strong>Tanggal Lahir:</strong> ${activePatient.dob || '-'}</p>
      <p><strong>Umur:</strong> ${activePatient.age||computeAge(activePatient.dob)||'–'}</p>
      <p><strong>Alamat:</strong> ${activePatient.address}</p>
      <p><strong>Tingkat Kecocokan:</strong> ${activePatient.confidence ?? '-'}%</p>
    `;
    modalVerifDetail.classList.remove('hidden');
  });
  btnModalVerifTutup.addEventListener('click',()=>modalVerifDetail.classList.add('hidden'));
  btnModalVerifCloseX.addEventListener('click',()=>modalVerifDetail.classList.add('hidden'));
  btnModalVerifLanjut.addEventListener('click',()=>{
    modalVerifDetail.classList.add('hidden');
    if(!activePatient){showAlert('Data pasien tidak tersedia.');return;}
    gwNama.textContent=activePatient.name;
    gwUmur.textContent=activePatient.age||computeAge(activePatient.dob)||'–';
    gwAlamat.textContent=activePatient.address;
    showPage('page-poli-gateway');
  });

  // Manual NIK
  btnNikFallback.addEventListener('click',()=>{verifNikBox.classList.remove('hidden');verifResult.classList.add('hidden');});
  btnCariNik.addEventListener('click',async ()=>{
    const nik=fallbackNik.value.trim();
    if(!/^\d{16}$/.test(nik)){showAlert('Masukkan NIK 16 digit.');return;}
    showLoading('Cari pasien...');
    try{
      const r=await fetch(`/api/patient/${nik}`); const d=await r.json(); hideLoading();
      if(!d.ok){showAlert(d.msg||'NIK tidak ditemukan.');return;}
      activePatient={nik:d.patient.nik,name:d.patient.name,address:d.patient.address,dob:d.patient.dob,age:d.patient.age};
      verifData.innerHTML=`
        <p><strong>NIK:</strong> <span class="font-mono">${activePatient.nik}</span></p>
        <p><strong>Nama:</strong> ${activePatient.name}</p>
        <p><strong>Umur:</strong> ${activePatient.age||computeAge(activePatient.dob)||'–'}</p>
        <p><strong>Alamat:</strong> ${activePatient.address}</p>
      `;
      verifResult.classList.remove('hidden');
      showAlert('Data pasien ditemukan.');
    }catch(err){hideLoading();showAlert('Error: '+err.message);}
  });

  // Form Poli (queue assign)
  formPoliGateway.addEventListener('submit',async e=>{
    e.preventDefault();
    if(!activePatient){showAlert('Data pasien tidak tersedia.');return;}
    const poli=gwPoli.value; if(!poli){showAlert('Pilih poli.');return;}
    showLoading('Mengambil nomor antrian...');
    try{
      const r=await fetch('/api/queue/assign',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({poli})});
      const d=await r.json(); hideLoading();
      if(!d.ok){showAlert(d.msg||'Gagal ambil nomor.');return;}
      openAntrian(d.poli,d.nomor);
      e.target.reset(); activePatient=null;
    }catch(err){hideLoading();showAlert('Error jaringan: '+err.message);}
  });

  // Nav hooks
  navHome.addEventListener('click',()=>showPage('page-home'));
  navRegistrasi.addEventListener('click',()=>showPage('page-registrasi'));
  navPoli.addEventListener('click',()=>showPage('page-poli'));
  btnHomeRegistrasi.addEventListener('click',()=>showPage('page-registrasi'));
  btnHomeKePoli.addEventListener('click',()=>showPage('page-poli'));

  btnAlertOk.addEventListener('click', hideLoading); // OK juga nutup loading jika masih kebuka
  btnAlertOk.addEventListener('click', ()=>modalAlert.classList.add('hidden'));
  btnAntrianTutup.addEventListener('click',()=>{closeAntrian();showPage('page-home');});

  // Init
  showPage('page-home');
})();